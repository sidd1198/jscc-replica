"""train.py - train the JSCC codec. --lam 0 = plain MSE; --lam > 0 = inverse-area weighted."""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import data as D
from loss import build_weight_map, weighted_mse
from model import JSCC


class CropSet(Dataset):
    """Random crop x crop patches plus their weight maps."""

    def __init__(self, root, split, crop, lam, aref, gamma, limit=0):
        D.ROOT = Path(root)
        self.boxes = D.load_boxes(split)
        self.names = sorted(self.boxes)
        if limit:
            self.names = self.names[:limit]
        self.dir = Path(root) / f"VisDrone2019-DET-{split}" / "images"
        self.crop, self.lam, self.aref, self.gamma = crop, lam, aref, gamma

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        im = Image.open(self.dir / f"{name}.jpg").convert("RGB")
        if im.width < self.crop or im.height < self.crop:      # rare: pad small images
            canvas = Image.new("RGB", (max(im.width, self.crop), max(im.height, self.crop)))
            canvas.paste(im, (0, 0))
            im = canvas
        left = random.randint(0, im.width - self.crop)
        top = random.randint(0, im.height - self.crop)
        arr = np.array(im.crop((left, top, left + self.crop, top + self.crop)))
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255
        w = build_weight_map(self.boxes[name], left, top, self.crop, self.lam, self.aref, self.gamma)
        return x, torch.from_numpy(w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--aref", type=float, default=1024.0)
    p.add_argument("--c", type=int, default=24)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="use only N images (smoke test)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="runs/codec")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| lam:", a.lam, "| c:", a.c, "| epochs:", a.epochs)

    ds = CropSet(a.data_root, "train", a.crop, a.lam, a.aref, a.gamma, a.limit)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.num_workers, pin_memory=(device == "cuda"))
    model = JSCC(a.c).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck, start = out / "last.pt", 0
    if ck.exists():                                   # resume after a Colab disconnect
        s = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        start = s["epoch"]
        print(f"resumed from epoch {start}")

    for ep in range(start, a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, w in dl:
            x, w = x.to(device), w.to(device)
            snr = random.uniform(0, 20)               # one SNR per batch, as in the paper
            loss = weighted_mse(x, model(x, snr), w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item(), n + 1
        sched.step()
        print(f"epoch {ep + 1}/{a.epochs}  loss={tot / n:.5f}  {time.time() - t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep + 1, "c": a.c,
                    "args": vars(a)}, ck)
    torch.save({"model": model.state_dict(), "c": a.c, "args": vars(a)}, out / "final.pt")
    print("done ->", out / "final.pt")


if __name__ == "__main__":
    main()