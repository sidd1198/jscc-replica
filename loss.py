"""loss.py - inverse-area weight map and weighted MSE (paper eqs. 5-6)."""
import numpy as np
import torch


def build_weight_map(boxes, left, top, crop, lam, aref=1024.0, gamma=0.5, min_keep=0.3):
    """Weight map for one crop. Small boxes get large weight; mean is normalised to 1.

    boxes: [(x, y, w, h, cat), ...] in full-image pixels; (left, top) = crop origin.
    Boxes that keep under 30% of their area inside the crop are ignored.
    """
    s = np.zeros((crop, crop), np.float32)
    if lam > 0:
        for x, y, w, h, _ in boxes:
            area = w * h
            if area <= 0:
                continue
            x1, y1 = max(x, left), max(y, top)
            x2, y2 = min(x + w, left + crop), min(y + h, top + crop)
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) * (y2 - y1) < min_keep * area:
                continue
            s[y1 - top:y2 - top, x1 - left:x2 - left] += (aref / area) ** gamma
    wmap = 1.0 + lam * s
    return wmap / wmap.mean()


def weighted_mse(x, xhat, wmap):
    """x, xhat: (B,3,H,W); wmap: (B,H,W)."""
    return (wmap.unsqueeze(1) * (x - xhat) ** 2).mean()


if __name__ == "__main__":
    # Test 1: lam = 0 must reproduce plain MSE exactly
    x, xh = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    w0 = torch.from_numpy(np.stack([build_weight_map([(10, 10, 8, 7, 1)], 0, 0, 64, 0.0)] * 2))
    print("test 1:", torch.allclose(weighted_mse(x, xh, w0), torch.mean((x - xh) ** 2)))

    # Test 2: a 8x7 box with lam=4 should weigh about 18x the background
    w = build_weight_map([(10, 10, 8, 7, 1)], 0, 0, 64, 4.0)
    print(f"test 2: inside/outside ratio = {w[12, 12] / w[40, 40]:.1f} (expect about 18.1), mean = {w.mean():.3f}")

    # Test 3: a box mostly outside the crop is dropped (map stays flat)
    w = build_weight_map([(-100, 10, 110, 7, 1)], 0, 0, 64, 4.0)
    print("test 3: dropped box ->", bool(np.allclose(w, 1.0)))