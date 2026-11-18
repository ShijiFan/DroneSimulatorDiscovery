"""W12 shared pieces: the W8 negative chain applied to an object echo, and the frozen W8 detector."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

PRF, N, FWHM = 17e3, 256, 9e-3
LAM = 299_792_458.0 / 77e9
HERE = Path(__file__).resolve().parent
NET = HERE.parent / "runs" / "E3w8_posv2_20261013_061950" / "net_N2_body_plus_oscillators.pt"


def chain(xg: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """xg [B,N] complex128 object echo in its own frame -> bulk radial velocity, 9 ms beam, SNR 0-40 dB, unit power.
    Same draws as the W8 negatives (v_r U(-5,5) m/s, beam centre 7.5 +- 1 ms, SNR U(0,40) dB)."""
    B, dev = xg.shape[0], xg.device
    t = torch.arange(N, device=dev, dtype=torch.float64) / PRF
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=g, device=dev, dtype=torch.float64)
    v_r = u(-5, 5)
    carrier = torch.exp(1j * 2 * math.pi * (2 * v_r / LAM)[:, None] * t)
    tc = 7.5e-3 + u(-1e-3, 1e-3)
    w = torch.exp(-4 * math.log(2) * (t[None, :] - tc[:, None]) ** 2 / FWHM ** 2)
    sig = xg * carrier * w
    p = (sig.abs() ** 2).mean(1).clamp_min(1e-30)
    snr = u(0, 40)
    cr = torch.complex(torch.randn(B, N, generator=g, device=dev, dtype=torch.float64),
                       torch.randn(B, N, generator=g, device=dev, dtype=torch.float64))
    x = sig + torch.sqrt(p / 10 ** (snr / 10) / 2)[:, None] * cr
    return (x / torch.sqrt((x.abs() ** 2).mean(1, keepdim=True))).to(torch.complex64)


def load_detector(dev="cpu"):
    from train_e0b import Net
    net = Net().to(dev)
    net.load_state_dict(torch.load(NET, map_location=dev))
    net.eval()
    return net


def generate(fn, n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / PRF
    out = np.empty((n, N), np.complex128)
    for i in range(n):
        x = np.asarray(fn(t.copy(), LAM, rng), dtype=complex).ravel()
        if x.shape != (N,) or not np.all(np.isfinite(x)) or np.abs(x).max() == 0:
            raise ValueError(f"nontarget_signal must return {N} finite, non-zero complex samples; got shape {x.shape}")
        out[i] = x
    return out
