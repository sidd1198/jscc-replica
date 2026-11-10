"""evaluate.py - AP at IoU 0.5 for tiny / small / large objects (clean images)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from data import load_boxes

# Bins in px^2 (fixed before looking at results, as in the paper)
BINS = {
    "tiny": (0, 256),
    "small": (256, 1024),
    "large": (1024, float("inf")),
    "all": (0, float("inf")),
}
IOU_T = 0.5
VAL_IMG_DIR = Path("data/VisDrone2019-DET-val/images")


def iou_one_to_many(box, boxes):
    """IoU between one xyxy box and an (N,4) array of xyxy boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def match_image(dets, conf, cls, gt_list):
    """Match detections to ALL ground truth of the same class (no bin filtering).

    Returns one record per detection: (class, confidence, matched GT area or nan,
    detection area). The bin is decided later, from the area of the matched object.
    """
    g = np.array([(x, y, x + w, y + h) for x, y, w, h, c in gt_list], float).reshape(-1, 4)
    gcls = np.array([c - 1 for *_, c in gt_list], int)  # VisDrone cats 1..10 -> 0..9
    garea = np.array([w * h for _, _, w, h, _ in gt_list], float)
    taken = np.zeros(len(g), bool)

    records = []
    for i in np.argsort(-conf):  # highest confidence first
        d = dets[i]
        darea = (d[2] - d[0]) * (d[3] - d[1])
        matched_area = np.nan
        cand = np.where((gcls == cls[i]) & ~taken)[0]
        if len(cand):
            ious = iou_one_to_many(d, g[cand])
            j = ious.argmax()
            if ious[j] >= IOU_T:
                taken[cand[j]] = True
                matched_area = garea[cand[j]]
        records.append((cls[i], conf[i], matched_area, darea))
    return records


def ap_101(is_tp, n_gt):
    """101-point interpolated AP. is_tp is sorted by descending confidence."""
    if n_gt == 0:
        return None
    if len(is_tp) == 0:
        return 0.0
    tp = np.cumsum(is_tp)
    fp = np.cumsum(~is_tp)
    recall = tp / n_gt
    precision = tp / (tp + fp)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # make non-increasing
    total = 0.0
    for t in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, t, side="left")
        total += precision[idx] if idx < len(precision) else 0.0
    return total / 101


def ap_by_bin(records, gts):
    rec = np.array(records, float).reshape(-1, 4)
    gt_cls = np.array([c - 1 for boxes in gts.values() for *_, c in boxes], int)
    gt_area = np.array([w * h for boxes in gts.values() for _, _, w, h, _ in boxes], float)

    out = {}
    for bname, (lo, hi) in BINS.items():
        aps, n_total = [], 0
        for c in range(10):
            n_gt = int(np.sum((gt_cls == c) & (gt_area >= lo) & (gt_area < hi)))
            n_total += n_gt
            r = rec[rec[:, 0] == c]
            r = r[np.argsort(-r[:, 1])]
            matched = ~np.isnan(r[:, 2])
            tp = matched & (r[:, 2] >= lo) & (r[:, 2] < hi)     # matched an in-bin object
            fp = ~matched & (r[:, 3] >= lo) & (r[:, 3] < hi)    # unmatched, in-bin size
            ap = ap_101(tp[tp | fp], n_gt)   # detections matched to out-of-bin objects are ignored
            if ap is not None:
                aps.append(ap)
        out[bname] = (float(np.mean(aps)) if aps else 0.0, n_total)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--limit", type=int, default=0, help="only use the first N images (quick test)")
    args = p.parse_args()

    gts = load_boxes("val")
    names = sorted(gts)
    if args.limit:
        names = names[: args.limit]
    paths = [str(VAL_IMG_DIR / f"{n}.jpg") for n in names]
    gts = {n: gts[n] for n in names}

    model = YOLO(args.weights)
    records, t0 = [], time.time()
    stream = model.predict(source=paths, imgsz=args.imgsz, conf=0.001, iou=0.7,
                           max_det=300, stream=True, verbose=False)
    for k, (name, r) in enumerate(zip(names, stream)):
        b = r.boxes
        records += match_image(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                               b.cls.cpu().numpy().astype(int), gts[name])
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(names)} images, {time.time() - t0:.0f}s")

    res = ap_by_bin(records, gts)
    print(f"\nimages: {len(names)}   imgsz: {args.imgsz}")
    print(f"{'bin':<8}{'GT boxes':>10}{'AP@0.5':>10}")
    for bname, (ap, n) in res.items():
        print(f"{bname:<8}{n:>10}{ap:>10.4f}")
    print("\npaper (960 px, 40 epochs): tiny 0.0539, small 0.2306, large 0.4695")

    Path("results").mkdir(exist_ok=True)
    with open(f"results/clean_ceiling_{args.imgsz}.json", "w") as f:
        json.dump({k: {"ap": v[0], "n_gt": v[1]} for k, v in res.items()}, f, indent=2)


if __name__ == "__main__":
    main()