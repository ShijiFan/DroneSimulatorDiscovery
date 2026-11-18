"""Forward model v3 (E3w W7 follow-up): domain randomisation around the 4-parameter family fitted to real SAAB drone
spectra (runs/E3w7_20261013_004649/v3_family: r 3.6-6.8 mm, gamma 0.82, full-length line at -6.7 dB).
Per rotor: K=12 short-radius point scatterers, radii log-uniform in [r_min, r_max] (absolute, independent of L), amplitude
asymmetry exp(gamma sin(psi)/2); plus full-length Martin-Mulgrew blades (same asymmetry) at weight a_spec_db.
The size-asymmetry relation seen in W4 is deliberately NOT built in (gamma independent of L), so any L information the
net extracts on real data must come from the aliased full-blade component.
Body, beam, noise as sim_saab.render.
"""

from __future__ import annotations

import math

import torch

import sim_saab as S

K_PTS = 12
PRIOR3 = {
    "log10_r_min": (-3.0, -2.3),   # 1-5 mm
    "log10_r_span": (0.05, 0.6),   # r_max = r_min * 10^span
    "gamma": (0.2, 1.6),
    "a_spec_db": (-18.0, 3.0),
    "body_db": (0.0, 20.0),
}


def sample_drones_v3(B, g, device):
    d = S.sample_drones(B, g, device)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=g, device=device)
    for k in ("log10_r_min", "log10_r_span", "gamma", "a_spec_db", "body_db"):
        d[k] = u(*PRIOR3[k])
    return d


def render_v3(d, snr_db, g, device, drift=True):
    B = d["L"].shape[0]
    N, PRF, LAM = S.N, S.PRF, S.LAM
    t = torch.arange(N, device=device, dtype=torch.float64) / PRF
    look = 1 + (S.PRIOR["look_drift"] * (2 * torch.rand(B, generator=g, device=device) - 1) if drift else 0)
    f_r = ((d["f0"] * look)[:, None] * d["spread"]).double()  # [B,6]
    phi = 2 * math.pi * torch.rand(B, 6, generator=g, device=device, dtype=torch.float64)
    cb = torch.cos(d["beta"]).double()
    gam = d["gamma"].double()
    rot_mask = (torch.arange(6, device=device)[None, :] < d["n_rot"][:, None]).double()
    kk = 4 * math.pi / LAM

    # short-radius scatterers
    lr0 = d["log10_r_min"].double()[:, None, None]
    lr1 = lr0 + d["log10_r_span"].double()[:, None, None]
    u = torch.rand(B, 6, K_PTS, generator=g, device=device, dtype=torch.float64)
    r = 10 ** (lr0 + (lr1 - lr0) * u)  # [B,6,K]
    ph = 2 * math.pi * torch.rand(B, 6, K_PTS, generator=g, device=device, dtype=torch.float64)
    amp = torch.complex(torch.randn(B, 6, K_PTS, generator=g, device=device, dtype=torch.float64),
                        torch.randn(B, 6, K_PTS, generator=g, device=device, dtype=torch.float64)) / math.sqrt(2)
    psi = 2 * math.pi * f_r[:, :, None, None] * t + phi[:, :, None, None] + ph[..., None]  # [B,6,K,N]
    pts = (amp[..., None] * rot_mask[:, :, None, None] * torch.exp(0.5 * gam[:, None, None, None] * torch.sin(psi))
           * torch.exp(1j * kk * (r * cb[:, None, None])[..., None] * torch.cos(psi))).sum((1, 2))

    # full-length blades
    k3 = torch.arange(3, device=device)
    blade_off = (2 * math.pi * k3[None, :] / d["n_b"][:, None]).double()
    psib = 2 * math.pi * f_r[:, :, None, None] * t + phi[:, :, None, None] + blade_off[:, None, :, None]  # [B,6,3,N]
    bl_mask = (k3[None, :] < d["n_b"][:, None]).double()
    m6 = rot_mask[:, :, None, None] * bl_mask[:, None, :, None]
    x = kk * d["L_eff"].double()[:, None, None, None] * torch.cos(psib)
    line = (torch.exp(0.5j * x) * torch.sinc(x / (2 * math.pi)) * m6
            * torch.exp(0.5 * gam[:, None, None, None] * torch.sin(psib))).sum((1, 2))

    pw = lambda z: (z.abs() ** 2).mean(1).clamp_min(1e-30)
    rotor = pts + line * torch.sqrt(pw(pts) / pw(line) * 10 ** (d["a_spec_db"].double() / 10))[:, None]

    f_body = 2 * d["v_r"].double() / LAM
    carrier = torch.exp(1j * (2 * math.pi * f_body[:, None] * t))
    tc = 7.5e-3 + 1e-3 * (2 * torch.rand(B, generator=g, device=device).double() - 1)
    w = torch.exp(-4 * math.log(2) * (t[None, :] - tc[:, None]) ** 2 / S.FWHM ** 2)
    rotor = rotor * carrier * w
    p_rot = pw(rotor)
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
