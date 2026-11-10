"""channel.py - power normalisation + noisy channel (AWGN / Rayleigh / Rician).

Input z: real tensor (B, C, H, W) from the encoder. We treat it as k = C*H*W/2
complex symbols per image (first half = real parts, second half = imaginary).
"""
import math
import torch


def to_complex(z):
    """(B,C,H,W) real -> (B,k) complex."""
    flat = z.reshape(z.shape[0], -1).float()
    assert flat.shape[1] % 2 == 0, "C*H*W must be even"
    k = flat.shape[1] // 2
    return torch.complex(flat[:, :k], flat[:, k:])


def from_complex(zc, shape):
    """(B,k) complex -> real tensor of the given shape."""
    return torch.cat([zc.real, zc.imag], dim=1).reshape(shape)


def power_normalize(zc, P=1.0):
    """Scale each image so the average |symbol|^2 equals P (paper eq. 1)."""
    k = zc.shape[1]
    energy = (zc.abs() ** 2).sum(dim=1, keepdim=True)
    return zc * torch.sqrt(k * P / (energy + 1e-12))


def fading_coeff(kind, B, device, K_dB=10.0):
    """One complex gain h per image (block fading). AWGN -> h = 1."""
    if kind == "awgn":
        return torch.ones(B, 1, dtype=torch.complex64, device=device)
    cn = torch.complex(torch.randn(B, 1, device=device),
                       torch.randn(B, 1, device=device)) / math.sqrt(2)  # CN(0,1)
    if kind == "rayleigh":
        return cn
    if kind == "rician":
        K = 10 ** (K_dB / 10)
        return math.sqrt(K / (K + 1)) + math.sqrt(1 / (K + 1)) * cn
    raise ValueError(f"unknown channel: {kind}")


def channel(z, snr_db, kind="awgn", K_dB=10.0):
    """Normalise -> fade -> add noise -> equalise (divide by h). Differentiable.

    snr_db: a float, or a tensor of shape (B,) with one SNR per image.
    Always runs in float32, even under mixed-precision training.
    """
    with torch.autocast(device_type=z.device.type, enabled=False):
        shape = z.shape
        zc = power_normalize(to_complex(z))
        B, k = zc.shape
        snr = torch.as_tensor(snr_db, dtype=torch.float32, device=z.device)
        snr = snr.expand(B).reshape(B, 1)
        sigma2 = 10 ** (-snr / 10)                       # noise power
        noise = torch.sqrt(sigma2 / 2) * torch.complex(
            torch.randn(B, k, device=z.device), torch.randn(B, k, device=z.device))
        h = fading_coeff(kind, B, z.device, K_dB)
        y = h * zc + noise
        y = y / h                                        # zero-forcing equaliser
        return from_complex(y, shape)


if __name__ == "__main__":
    torch.manual_seed(0)
    z = torch.randn(16, 24, 48, 48)          # like encoder output: c=24, 384/8=48
    zn = from_complex(power_normalize(to_complex(z)), z.shape)

    # Test 1: average power per complex symbol should be 1
    p = (to_complex(zn).abs() ** 2).mean().item()
    print(f"test 1: signal power = {p:.4f}   (expect 1.0000)")

    # Test 2: measured SNR should match the requested SNR (AWGN)
    for snr in (0, 10, 20):
        out = channel(z, snr, "awgn")
        noise = out - zn
        measured = 10 * math.log10((zn ** 2).mean().item() / (noise ** 2).mean().item())
        print(f"test 2: asked {snr:>2} dB -> measured {measured:6.2f} dB")

    # Test 3: Rician should come out roughly 1 dB below nominal
    out = channel(z, 10, "rician")
    measured = 10 * math.log10((zn ** 2).mean().item() / ((out - zn) ** 2).mean().item())
    print(f"test 3: rician asked 10 dB -> measured {measured:.2f} dB (paper: about 1 dB lower)")

    # Test 4: gradients must flow back through the channel
    z.requires_grad_(True)
    (channel(z, 10, "awgn") ** 2).mean().backward()
    print("test 4: gradient flows:", z.grad is not None and z.grad.abs().sum().item() > 0)