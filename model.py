"""model.py - SNR-adaptive JSCC encoder/decoder (paper Fig. 2)."""
import torch
import torch.nn as nn

from channel import channel


class AF(nn.Module):
    """Attention-feature module: gates each channel using pooled features + SNR."""

    def __init__(self, C):
        super().__init__()
        self.fc1 = nn.Linear(C + 1, C // 4)
        self.fc2 = nn.Linear(C // 4, C)

    def forward(self, u, snr):                       # u: (B,C,h,w), snr: (B,) in dB
        s = u.mean(dim=(2, 3))                       # global average pool -> (B,C)
        s = torch.cat([s, (snr / 20).unsqueeze(1)], dim=1)   # add normalised SNR
        g = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))  # per-channel gain
        return u * g[:, :, None, None]


class Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.a1 = nn.Conv2d(3, 64, 5, 2, 2), nn.PReLU()
        self.c2, self.a2 = nn.Conv2d(64, 128, 5, 2, 2), nn.PReLU()
        self.af1 = AF(128)
        self.c3, self.a3 = nn.Conv2d(128, 128, 5, 2, 2), nn.PReLU()
        self.af2 = AF(128)
        self.proj = nn.Conv2d(128, c, 3, 1, 1)

    def forward(self, x, snr):
        x = self.a1(self.c1(x))
        x = self.a2(self.c2(x))
        x = self.af1(x, snr)
        x = self.a3(self.c3(x))
        x = self.af2(x, snr)
        return self.proj(x)                          # (B, c, H/8, W/8)


class Decoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c0, self.a0 = nn.Conv2d(c, 128, 3, 1, 1), nn.PReLU()
        self.af1 = AF(128)
        self.d1, self.a1 = nn.ConvTranspose2d(128, 128, 5, 2, 2, output_padding=1), nn.PReLU()
        self.af2 = AF(128)
        self.d2, self.a2 = nn.ConvTranspose2d(128, 64, 5, 2, 2, output_padding=1), nn.PReLU()
        self.d3 = nn.ConvTranspose2d(64, 3, 5, 2, 2, output_padding=1)

    def forward(self, y, snr):
        x = self.a0(self.c0(y))
        x = self.af1(x, snr)
        x = self.a1(self.d1(x))
        x = self.af2(x, snr)
        x = self.a2(self.d2(x))
        return torch.sigmoid(self.d3(x))             # (B, 3, H, W) in [0, 1]


class JSCC(nn.Module):
    """image -> encoder -> noisy channel -> decoder -> reconstructed image."""

    def __init__(self, c=24):
        super().__init__()
        self.enc, self.dec = Encoder(c), Decoder(c)

    def forward(self, x, snr_db, kind="awgn"):
        snr = torch.as_tensor(snr_db, dtype=torch.float32, device=x.device).expand(x.shape[0])
        z = self.enc(x, snr)
        y = channel(z, snr, kind)
        return self.dec(y, snr)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 3, 256, 256)
    for c in (12, 24, 48):
        m = JSCC(c)
        out = m(x, 10.0)
        print(f"c={c:<2}  R=c/384={c / 384:.4f}  enc={n_params(m.enc) / 1e6:.3f}M  "
              f"dec={n_params(m.dec) / 1e6:.3f}M  out={tuple(out.shape)}  "
              f"range=[{out.min():.2f}, {out.max():.2f}]")
        assert out.shape == x.shape
    # one SNR per image, a non-square input, and a fading channel all work
    m = JSCC(24)
    print("per-image SNR :", tuple(m(x, torch.tensor([0.0, 20.0])).shape))
    print("non-square    :", tuple(m(torch.rand(1, 3, 192, 320), 10.0, "rayleigh").shape))