"""Forward model v2 (PROTOCOL.md E3a / E3w W2): domain-randomised multirotor slow-time IQ for SAAB-like conditions.

Motivated by runs/E3w1_20261013_002937/REPORT_ZH.md: real drone residuals show (1) a low-Doppler hump within
~2.5 kHz of the body line, (2) a flat floor ~12 dB above noise, (3) 2-6 dB negative/positive asymmetry; v1 had only (2).
Components per rotor (each drone draws its own mixture weights, so the net must not rely on any single one):
  spec   coherent Martin-Mulgrew line (v1), weight a_spec
  diff   M random-phase point scatterers per blade, radii r = r0 + (L-r0)*u^(1+alpha) (alpha>0 -> towards the root),
         amplitude x exp(gamma*sin(psi)/2) for advancing/retreating asymmetry
  hub    K points per rotor at radius r_h (motor bell / hub / screws), same rotation rate
Body: tone x beam, plus sinusoidal vibration (amplitude a_v, freq f_v) and optional radar spur sidebands.
Labels stay L_eff = L cos(beta) and f0 (rotation rate).
"""

from __future__ import annotations

import math

import torch

import sim_saab as S

M_DIFF = 8
K_HUB = 4

PRIOR2 = {
    "a_spec_db": (-35.0, 0.0),     # coherent line weight relative to diffuse blade power
    "alpha": (0.0, 6.0),
    "gamma": (0.0, 2.0),          # fixed sign: approaching blade shows its leading edge (sign fitted to SAAB IQ convention)
    "gamma_hub": (0.0, 3.0),       # same leading-edge asymmetry for low-radius (non-aliased) scatterers
    "hub_db": (-15.0, 10.0),       # hub power relative to diffuse blade power
    "r_hub": (0.005, 0.035),
    "body_db": (-5.0, 25.0),       # body power relative to total rotor power
    "vib_amp": (0.0, 3e-4),        # m
    "vib_f": (20.0, 400.0),        # Hz
    "spur_p": 0.5,
    "spur_f": (3e3, 6e3),
    "spur_db": (-40.0, -20.0),
}


def sample_drones_v2(B, g, device):
    d = S.sample_drones(B, g, device)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=g, device=device)
    for k in ("a_spec_db", "alpha", "gamma", "gamma_hub", "hub_db", "r_hub", "body_db", "vib_amp", "vib_f", "spur_f", "spur_db"):
        d[k] = u(*PRIOR2[k])
    d["spur_on"] = (torch.rand(B, generator=g, device=device) < PRIOR2["spur_p"]).double()
    d["vib_f"] = torch.where(torch.rand(B, generator=g, device=device) < 0.5, d["f0"], d["vib_f"])  # half at rotor rate
    return d


def _cplx_randn(shape, g, device):
    return torch.complex(torch.randn(shape, generator=g, device=device, dtype=torch.float64),
                         torch.randn(shape, generator=g, device=device, dtype=torch.float64)) / math.sqrt(2)


def render_v2(d, snr_db, g, device, drift=True, return_parts=False):
    B = d["L"].shape[0]
    N, PRF, LAM = S.N, S.PRF, S.LAM
    t = torch.arange(N, device=device, dtype=torch.float64) / PRF
    look = 1 + (S.PRIOR["look_drift"] * (2 * torch.rand(B, generator=g, device=device) - 1) if drift else 0)
    f_r = ((d["f0"] * look)[:, None] * d["spread"]).double()  # [B,6]
    phi = 2 * math.pi * torch.rand(B, 6, generator=g, device=device, dtype=torch.float64)
    k = torch.arange(3, device=device)
    blade_off = (2 * math.pi * k[None, :] / d["n_b"][:, None]).double()
    psi = 2 * math.pi * f_r[:, :, None, None] * t + phi[:, :, None, None] + blade_off[:, None, :, None]  # [B,6,3,N]
    cb = torch.cos(d["beta"]).double()
    rot_mask = (torch.arange(6, device=device)[None, :] < d["n_rot"][:, None]).double()
    bl_mask = (k[None, :] < d["n_b"][:, None]).double()
    m6 = rot_mask[:, :, None, None] * bl_mask[:, None, :, None]

    # coherent line (v1)
    x = (4 * math.pi / LAM) * d["L_eff"].double()[:, None, None, None] * torch.cos(psi)
    spec = (torch.exp(0.5j * x) * torch.sinc(x / (2 * math.pi)) * m6
            * torch.exp(0.5 * d["gamma"].double()[:, None, None, None] * torch.sin(psi))).sum((1, 2))

    # diffuse blade points
    L = d["L"].double()
    r0 = 0.15 * L
    u = torch.rand(B, 6, 3, M_DIFF, generator=g, device=device, dtype=torch.float64)
    r = r0[:, None, None, None] + (L - r0)[:, None, None, None] * u ** (1 + d["alpha"].double()[:, None, None, None])
    amp = _cplx_randn((B, 6, 3, M_DIFF), g, device)
    asym = torch.exp(0.5 * d["gamma"].double()[:, None, None, None] * torch.sin(psi))  # [B,6,3,N], power ratio e^(2 gamma)
    diff = torch.zeros(B, N, dtype=torch.complex128, device=device)
    cpsi = torch.cos(psi) * m6
    for m in range(M_DIFF):
        ph = (4 * math.pi / LAM) * (r[..., m] * cb[:, None, None])[..., None] * cpsi
        diff = diff + (amp[..., m, None] * asym * m6 * torch.exp(1j * ph)).sum((1, 2))

    # hub points
    ph_h = 2 * math.pi * torch.rand(B, 6, K_HUB, generator=g, device=device, dtype=torch.float64)
    psi_h = 2 * math.pi * f_r[:, :, None, None] * t + ph_h[..., None]  # [B,6,K,N]
    amp_h = _cplx_randn((B, 6, K_HUB), g, device)
    rh = (d["r_hub"].double() * cb)[:, None, None, None]
    hub = (amp_h[..., None] * rot_mask[:, :, None, None] * torch.exp(1j * (4 * math.pi / LAM) * rh * torch.cos(psi_h))
           * torch.exp(0.5 * d["gamma_hub"].double()[:, None, None, None] * torch.sin(psi_h))).sum((1, 2))

    pw = lambda z: (z.abs() ** 2).mean(1).clamp_min(1e-30)
    p_diff = pw(diff)
    spec = spec * torch.sqrt(p_diff / pw(spec) * 10 ** (d["a_spec_db"].double() / 10))[:, None]
    hub = hub * torch.sqrt(p_diff / pw(hub) * 10 ** (d["hub_db"].double() / 10))[:, None]
    rotor = spec + diff + hub

    f_body = 2 * d["v_r"].double() / LAM
    vib = (4 * math.pi / LAM) * d["vib_amp"].double()[:, None] * torch.sin(
        2 * math.pi * d["vib_f"].double()[:, None] * t + 2 * math.pi * torch.rand(B, 1, generator=g, device=device, dtype=torch.float64))
    carrier = torch.exp(1j * (2 * math.pi * f_body[:, None] * t))
    tc = 7.5e-3 + 1e-3 * (2 * torch.rand(B, generator=g, device=device).double() - 1)
    w = torch.exp(-4 * math.log(2) * (t[None, :] - tc[:, None]) ** 2 / S.FWHM ** 2)
    rotor = rotor * carrier * w
    p_rot = pw(rotor)
    body_amp = torch.sqrt(p_rot * 10 ** (d["body_db"].double() / 10))
    body_ph = 2 * math.pi * torch.rand(B, generator=g, device=device).double()
    spur = d["spur_on"].double()[:, None] * 10 ** (d["spur_db"].double()[:, None] / 20) * 2 * torch.cos(
        2 * math.pi * d["spur_f"].double()[:, None] * t)
    body = body_amp[:, None] * torch.exp(1j * (body_ph[:, None] + vib)) * (1 + spur) * carrier * w
    sigma2 = p_rot / 10 ** (snr_db.double() / 10)
    noise = torch.sqrt(sigma2 / 2)[:, None] * torch.complex(
        torch.randn(B, N, generator=g, device=device, dtype=torch.float64),
        torch.randn(B, N, generator=g, device=device, dtype=torch.float64))
    sig = rotor + body + noise
    sig = sig / torch.sqrt((sig.abs() ** 2).mean(1, keepdim=True))
    if return_parts:
        return sig.to(torch.complex64), {"spec": spec, "diff": diff, "hub": hub}
    return sig.to(torch.complex64)

