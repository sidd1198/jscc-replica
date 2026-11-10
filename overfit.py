"""overfit.py - can the model memorise 8 images? (sanity check before real training)"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import JSCC

SIZE, N, STEPS, SNR = 256, 8, 200, 20.0


def load_images():
    paths = sorted(Path("data/VisDrone2019-DET-val/images").glob("*.jpg"))[:N]
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if min(im.size) < SIZE:
            s = SIZE / min(im.size)
            im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1))
        l, t = (im.width - SIZE) // 2, (im.height - SIZE) // 2
        arr = np.asarray(im.crop((l, t, l + SIZE, t + SIZE)))
        out.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255)
    return torch.stack(out)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
x = load_images().to(device)
model = JSCC(24).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)   # higher lr than the real run, just for this test

for step in range(1, STEPS + 1):
    loss = torch.mean((model(x, SNR) - x) ** 2)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step == 1 or step % 25 == 0:
        print(f"step {step:>3}  mse={loss.item():.5f}  psnr={-10 * torch.log10(loss).item():.2f} dB")

model.eval()
with torch.no_grad():
    out = model(x, SNR)
Path("results").mkdir(exist_ok=True)
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    ax[0, i].imshow(x[i].cpu().permute(1, 2, 0)); ax[0, i].axis("off")
    ax[1, i].imshow(out[i].cpu().permute(1, 2, 0).clamp(0, 1)); ax[1, i].axis("off")
ax[0, 0].set_title("input (top) / reconstruction (bottom)", loc="left")
plt.savefig("results/overfit.png", dpi=100)
print("saved results/overfit.png")