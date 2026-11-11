"""eval_codec.py - AP by object size after: image -> codec -> noisy channel -> frozen YOLO.

The codec sees the full-resolution image (the scale it was trained at). The
reconstruction has the same size, so detections are already in original pixel
coordinates and are matched exactly as in evaluate.py.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

import data as D
from evaluate import match_image, ap_by_bin
from model import JSCC

BINS = ("tiny", "small", "large", "all")


def load_codec(path, device):
    s = torch.load(path, map_location=device, weights_only=False)
    m = JSCC(s["c"]).to(device)
    m.load_state_dict(s["model"])
    return m.eval()


@torch.no_grad()
def transmit(model, arr, snr, kind, device):
    """arr: HxWx3 uint8 RGB -> reconstruction of the same shape and type."""
    x = torch.from_numpy(arr).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    H, W = x.shape[2:]
    x = F.pad(x, (0, (-W) % 8, 0, (-H) % 8), mode="replicate")  # encoder needs multiples of 8
    y = model(x, snr, kind)[:, :, :H, :W]
    return (y.clamp(0, 1) * 255).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="codec checkpoint (final.pt or last.pt)")
    p.add_argument("--det", default="weights/best.pt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--snrs", default="0,5,10,15,20")
    p.add_argument("--channel", default="awgn", choices=["awgn", "rician", "rayleigh"])
    p.add_argument("--imgsz", type=int, default=640, help="detector size, same as evaluate.py")
    p.add_argument("--limit", type=int, default=0, help="first N images only (quick test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/codec.json")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    snrs = [float(s) for s in a.snrs.split(",")]

    D.ROOT = Path(a.data_root)
    gts = D.load_boxes("val")
    names = sorted(gts)[: a.limit or None]
    gts = {n: gts[n] for n in names}
    img_dir = D.ROOT / "VisDrone2019-DET-val" / "images"

    codec = load_codec(a.ckpt, device)
    det = YOLO(a.det)
    records = {s: [] for s in snrs}
    t0 = time.time()
    for k, name in enumerate(names):
        arr = np.array(Image.open(img_dir / f"{name}.jpg").convert("RGB"))
        for s in snrs:
            rec = transmit(codec, arr, s, a.channel, device)
            r = det.predict(source=np.ascontiguousarray(rec[:, :, ::-1]),  # RGB -> BGR
                            imgsz=a.imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
            b = r.boxes
            records[s] += match_image(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                      b.cls.cpu().numpy().astype(int), gts[name])
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(names)} images, {time.time() - t0:.0f}s", flush=True)

    res = {s: ap_by_bin(records[s], gts) for s in snrs}
    ceil_path = Path(f"results/clean_ceiling_{a.imgsz}.json")
    ceil = json.loads(ceil_path.read_text()) if ceil_path.exists() else None

    print(f"\ncodec: {a.ckpt}   channel: {a.channel}   images: {len(names)}")
    print("AP@0.5 by object size")
    print(f"{'SNR':>5}" + "".join(f"{b:>9}" for b in BINS))
    for s in snrs:
        print(f"{s:>5.0f}" + "".join(f"{res[s][b][0]:>9.4f}" for b in BINS))
    if ceil:
        print("\n% of clean-image AP retained (meaningful on the full 548 images)")
        print(f"{'SNR':>5}" + "".join(f"{b:>9}" for b in BINS[:3]))
        for s in snrs:
            print(f"{s:>5.0f}" + "".join(f"{100 * res[s][b][0] / ceil[b]['ap']:>8.1f}%" for b in BINS[:3]))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": a.ckpt, "channel": a.channel, "images": len(names),
                               "ap": {str(int(s)): {b: res[s][b][0] for b in BINS} for s in snrs}},
                              indent=2))
    fig, ax = plt.subplots(figsize=(6, 4))
    for b in BINS[:3]:
        y = [100 * res[s][b][0] / ceil[b]["ap"] for s in snrs] if ceil else [res[s][b][0] for s in snrs]
        ax.plot(snrs, y, marker="o", label=b)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("AP retained vs clean (%)" if ceil else "AP@0.5")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out.with_suffix(".png"), dpi=120, bbox_inches="tight")
    print("saved", out, "and", out.with_suffix(".png"))


if __name__ == "__main__":
    main()