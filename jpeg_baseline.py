"""jpeg_baseline.py - separation baseline: JPEG + an ideal (capacity-achieving) channel code.

Per frame and SNR (paper Sec. V-F):
  channel uses   k      = R * 3*H*W          (R = c/384, so c=24 gives R = 1/16, same bandwidth as the codec)
  bit budget     bits   = k * log2(1 + SNR)  (under fading: SNR * |h|^2, drawn per frame)
  JPEG quality   = highest quality in 1..100 whose file fits the budget (binary search)
  if even quality 1 does not fit, the frame is lost: no detections for it.
The decoded JPEG goes to the frozen detector and is scored exactly like eval_codec.py.
"""
import argparse
import io
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

import data as D
from evaluate import match_image, ap_by_bin
from ultralytics import YOLO

BINS = ("tiny", "small", "large", "all")


def jpeg_bytes(img, q, cache):
    """JPEG-encode img at quality q (cached, so repeated SNRs reuse encodes)."""
    if q not in cache:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        cache[q] = buf.getvalue()
    return cache[q]


def best_quality(img, budget_bits, cache):
    """Highest JPEG quality (1..100) whose file fits budget_bits; None if even quality 1 overflows."""
    if 8 * len(jpeg_bytes(img, 1, cache)) > budget_bits:
        return None
    lo, hi = 1, 100                                   # quality lo is known to fit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if 8 * len(jpeg_bytes(img, mid, cache)) <= budget_bits:
            lo = mid
        else:
            hi = mid - 1
    return lo


def gain_sq(kind, rng, K_dB=10.0):
    """|h|^2 for one frame (block fading). AWGN -> 1."""
    if kind == "awgn":
        return 1.0
    cn = (rng.standard_normal() + 1j * rng.standard_normal()) / math.sqrt(2)   # CN(0,1)
    if kind == "rician":
        K = 10 ** (K_dB / 10)
        cn = math.sqrt(K / (K + 1)) + math.sqrt(1 / (K + 1)) * cn
    return abs(cn) ** 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det", default="weights/best.pt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--snrs", default="0,5,10,15,20")
    p.add_argument("--channel", default="awgn", choices=["awgn", "rician", "rayleigh"])
    p.add_argument("--c", type=int, default=24, help="same c as the codec you compare with; R = c/384")
    p.add_argument("--imgsz", type=int, default=640, help="detector size, same as evaluate.py")
    p.add_argument("--limit", type=int, default=0, help="first N images only (quick test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/jpeg.json")
    a = p.parse_args()

    R = a.c / 384
    rng = np.random.default_rng(a.seed)
    snrs = [float(s) for s in a.snrs.split(",")]
    D.ROOT = Path(a.data_root)
    gts = D.load_boxes("val")
    names = sorted(gts)[: a.limit or None]
    gts = {n: gts[n] for n in names}
    img_dir = D.ROOT / "VisDrone2019-DET-val" / "images"

    det = YOLO(a.det)
    records = {s: [] for s in snrs}
    lost = {s: 0 for s in snrs}
    qsum = {s: 0 for s in snrs}
    t0 = time.time()
    for k, name in enumerate(names):
        img = Image.open(img_dir / f"{name}.jpg").convert("RGB")
        W, H = img.size
        uses = R * 3 * H * W                                  # complex channel uses
        cache = {}
        for s in snrs:
            snr_eff = 10 ** (s / 10) * gain_sq(a.channel, rng)
            q = best_quality(img, uses * math.log2(1 + snr_eff), cache)
            if q is None:                                     # frame lost
                lost[s] += 1
                continue
            qsum[s] += q
            rec = np.array(Image.open(io.BytesIO(jpeg_bytes(img, q, cache))).convert("RGB"))
            r = det.predict(source=np.ascontiguousarray(rec[:, :, ::-1]),   # RGB -> BGR
                            imgsz=a.imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
            b = r.boxes
            records[s] += match_image(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                      b.cls.cpu().numpy().astype(int), gts[name])
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(names)} images, {time.time() - t0:.0f}s", flush=True)

    n = len(names)
    res = {s: ap_by_bin(records[s], gts) for s in snrs}
    ceil_path = Path(f"results/clean_ceiling_{a.imgsz}.json")
    ceil = json.loads(ceil_path.read_text()) if ceil_path.exists() else None

    print(f"\nJPEG + capacity code   channel: {a.channel}   R = {R:.4f}   images: {n}")
    print("AP@0.5 by object size")
    print(f"{'SNR':>5}" + "".join(f"{b:>9}" for b in BINS) + f"{'bpp':>8}{'lost%':>8}{'mean q':>8}")
    for s in snrs:
        bpp = R * 3 * math.log2(1 + 10 ** (s / 10))            # nominal budget, bits per pixel
        ok = n - lost[s]
        mq = qsum[s] / ok if ok else float("nan")
        print(f"{s:>5.0f}" + "".join(f"{res[s][b][0]:>9.4f}" for b in BINS)
              + f"{bpp:>8.3f}{100 * lost[s] / n:>8.1f}{mq:>8.1f}")
    if ceil:
        print("\n% of clean-image AP retained (meaningful on the full 548 images)")
        print(f"{'SNR':>5}" + "".join(f"{b:>9}" for b in BINS[:3]))
        for s in snrs:
            print(f"{s:>5.0f}" + "".join(f"{100 * res[s][b][0] / ceil[b]['ap']:>8.1f}%" for b in BINS[:3]))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": "jpeg+capacity", "channel": a.channel, "images": n, "R": R,
        "ap": {str(int(s)): {b: res[s][b][0] for b in BINS} for s in snrs},
        "lost_pct": {str(int(s)): 100 * lost[s] / n for s in snrs},
        "mean_quality": {str(int(s)): (qsum[s] / (n - lost[s]) if n - lost[s] else None) for s in snrs},
    }, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()