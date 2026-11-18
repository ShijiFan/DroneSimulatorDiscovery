"""Batched torch simulator of SAAB-SIRS-like slow-time IQ for multirotor UAVs (PROTOCOL.md E0b).

Conditions from E:/rag/datasets/data3/ReadMe.txt: 77 GHz, PRF 17 kHz, 256 azimuth samples (15 ms),
mechanical scan -> Gaussian two-way beam taper with FWHM 9 ms (effective dwell). Blade model: closed-form
Martin-Mulgrew line scatterer, as in rotor.py.
"""

from __future__ import annotations

import math

import torch

C = 299_792_458.0
FC = 77e9
LAM = C / FC
PRF = 17e3
N = 256
FWHM = 9e-3

PRIOR = {
    "L": (0.03, 0.25),
    "v_tip": (40.0, 150.0),
    "beta_deg": (0.0, 30.0),
    "v_r": (-5.0, 5.0),
    "body_db": (0.0, 20.0),
    "rotor_spread": 0.05,
    "look_drift": 0.02,
}


def sample_drones(B, g, device):
    """Per-drone parameters (constant across looks)."""
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=g, device=device)
    L = u(*PRIOR["L"])
    v_tip = u(*PRIOR["v_tip"])
    beta = torch.deg2rad(u(*PRIOR["beta_deg"]))
    f0 = v_tip / (2 * math.pi * L)
    n_rot = torch.where(torch.rand(B, generator=g, device=device) < 0.5, 4, 6)
    n_b = torch.where(torch.rand(B, generator=g, device=device) < 0.5, 2, 3)
    spread = 1 + PRIOR["rotor_spread"] * (2 * torch.rand(B, 6, generator=g, device=device) - 1)
    return {
        "L": L, "beta": beta, "L_eff": L * torch.cos(beta), "f0": f0, "n_rot": n_rot, "n_b": n_b,
        "spread": spread, "v_r": u(*PRIOR["v_r"]), "body_db": u(*PRIOR["body_db"]),
    }


def render(d, snr_db, g, device, drift=True):
    """One look per drone. snr_db: tensor [B] (rotor power after beam taper / noise power)."""
    B = d["L"].shape[0]
    t = torch.arange(N, device=device, dtype=torch.float64) / PRF
    look = 1 + (PRIOR["look_drift"] * (2 * torch.rand(B, generator=g, device=device) - 1) if drift else 0)
    f_r = (d["f0"] * look)[:, None] * d["spread"]  # [B,6]
    phi = 2 * math.pi * torch.rand(B, 6, generator=g, device=device)
    k = torch.arange(3, device=device)
    blade_off = 2 * math.pi * k[None, :] / d["n_b"][:, None]  # [B,3]
    psi = (2 * math.pi * f_r[:, :, None, None].double() * t
           + phi[:, :, None, None].double() + blade_off[:, None, :, None].double())  # [B,6,3,N]
    x = (4 * math.pi / LAM) * d["L_eff"][:, None, None, None].double() * torch.cos(psi)
    blade = torch.exp(0.5j * x) * torch.sinc(x / (2 * math.pi))
    rot_mask = (torch.arange(6, device=device)[None, :] < d["n_rot"][:, None]).double()
    bl_mask = (k[None, :] < d["n_b"][:, None]).double()
    rotor = (blade * rot_mask[:, :, None, None] * bl_mask[:, None, :, None]).sum((1, 2))  # [B,N]

    f_body = 2 * d["v_r"].double() / LAM
    carrier = torch.exp(1j * (2 * math.pi * f_body[:, None] * t))
    tc = 7.5e-3 + 1e-3 * (2 * torch.rand(B, generator=g, device=device).double() - 1)
    w = torch.exp(-4 * math.log(2) * (t[None, :] - tc[:, None]) ** 2 / FWHM ** 2)
    rotor = rotor * carrier * w
    p_rot = (rotor.abs() ** 2).mean(1)
    body_amp = torch.sqrt(p_rot * 10 ** (d["body_db"].double() / 10))
    body_ph = 2 * math.pi * torch.rand(B, generator=g, device=device).double()
    body = body_amp[:, None] * torch.exp(1j * body_ph[:, None]) * carrier * w
    sigma2 = p_rot / 10 ** (snr_db.double() / 10)
    noise = torch.sqrt(sigma2 / 2)[:, None] * torch.complex(
        torch.randn(B, N, generator=g, device=device, dtype=torch.float64),
        torch.randn(B, N, generator=g, device=device, dtype=torch.float64))
    sig = rotor + body + noise
    sig = sig / torch.sqrt((sig.abs() ** 2).mean(1, keepdim=True))
    return sig.to(torch.complex64)
