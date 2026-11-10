from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("data")
SKIP_CATS = {0, 11}  # ignored regions, "others"

def load_boxes(split):
    """Return {image_name: [(x, y, w, h, category), ...]}."""
    ann_dir = ROOT / f"VisDrone2019-DET-{split}" / "annotations"
    result = {}
    for f in sorted(ann_dir.glob("*.txt")):
        boxes = []
        for line in f.read_text().strip().splitlines():
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            x, y, w, h, score, cat = (int(v) for v in p[:6])
            if score == 0 or cat in SKIP_CATS:
                continue
            boxes.append((x, y, w, h, cat))
        result[f.stem] = boxes
    return result

if __name__ == "__main__":
    all_areas = []
    for split in ("train", "val"):
        data = load_boxes(split)
        areas = np.array([w * h for boxes in data.values() for (_, _, w, h, _) in boxes])
        all_areas.append(areas)
        n = len(areas)
        print(f"\n== {split} ==")
        print("images:", len(data))
        print("boxes :", n)
        print("median area:", np.median(areas))
        print(f"< 256     : {100 * (areas < 256).mean():.1f}%")
        print(f"256-1024  : {100 * ((areas >= 256) & (areas < 1024)).mean():.1f}%")
        print(f">= 1024   : {100 * (areas >= 1024).mean():.1f}%")
    plt.hist(np.concatenate(all_areas), bins=np.logspace(0, 5, 60))
    plt.xscale("log")
    plt.xlabel("box area (px^2)")
    plt.ylabel("count")
    plt.savefig("data/area_hist.png", dpi=120)
    print("\nsaved data/area_hist.png")