"""Paper figure: floor-referenced mean residual Doppler spectrum, real SAAB drones vs selected simulators
(same chain and estimator as the W9 evaluator e3w7_eval.py, seed 101)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

import e3w7_eval as EV
from w14_build_banks import load_fn

R = Path(__file__).resolve().parent.parent / "runs"
W9 = R / "W9_claude_20261013_124339"
MODELS = {
    "Textbook": R / "E3w7_20261013_004649/blind/iter_00_seed.py",
    "Karlsson": R / "W16_baselines/K_karlsson2023.py",
    "O-b2 (Opus)": W9 / "opus_blind_r2/best16.py",
    "S-b1 (Sonnet)": W9 / "sonnet_blind_r1/best.py",
}


def sim_shape(fn, seed=101):
    rng = np.random.default_rng(seed)
    t = np.arange(EV.N) / EV.PRF
    specs = []
    for _ in range(EV.N_DRONES):
        L = rng.uniform(0.03, 0.25)
        f0 = rng.uniform(40, 150) / (2 * np.pi * L)
        n_b = int(rng.choice([2, 3]))
        n_rot = int(rng.choice([4, 6]))
        beta = np.deg2rad(rng.uniform(0, 30))
        rot = np.asarray(fn(t.copy(), L, f0, n_b, n_rot, beta, EV.LAM, rng), dtype=complex).ravel()
        v_r = rng.uniform(-5, 5)
        carrier = np.exp(2j * np.pi * (2 * v_r / EV.LAM) * t)
        tc = 7.5e-3 + rng.uniform(-1e-3, 1e-3)
        w = np.exp(-4 * np.log(2) * (t - tc) ** 2 / EV.FWHM ** 2)
        rot = rot * carrier * w
        p = np.mean(np.abs(rot) ** 2)
        body = np.sqrt(p * 10 ** (rng.uniform(0, 20) / 10)) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * carrier * w
        noise = np.sqrt(p / 100 / 2) * (rng.standard_normal(EV.N) + 1j * rng.standard_normal(EV.N))
        specs.append(EV.spec(rot + body + noise))
    f = np.fft.fftshift(np.fft.fftfreq(EV.NF, 1 / EV.PRF))
    return f, EV.shape(np.mean(specs, 0), f)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    real = np.load(W9 / "real_shape.npz")
    fr, sr = real["f"], real["shape"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(fr / 1e3, sr, color="k", lw=2.2, label="Measured (mean of 6 drones)")
    styles = [("#999999", ":"), ("#1f77b4", "--"), ("#d62728", "-"), ("#ff7f0e", "-.")]
    meta = {}
    for (name, p), (c, ls) in zip(MODELS.items(), styles):
        f, s = sim_shape(load_fn(p))
        m = (np.abs(f) > 250) & (np.abs(f) < 8.5e3)
        rms = float(np.sqrt(np.mean((s[m] - sr[m]) ** 2)))
        ax.plot(f / 1e3, s, color=c, ls=ls, lw=1.1, label=f"{name}  (RMS {rms:.2f} dB)")
        meta[name] = {"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "rms_db_seed101": rms}
    ax.axvspan(-0.25, 0.25, color="0.9", zorder=0)
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-6, 30)
    ax.set_xlabel("Doppler frequency (kHz)")
    ax.set_ylabel("Residual level re floor (dB)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_spectra.png", dpi=600)
    fig.savefig(out / "fig_spectra.pdf")
    meta["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "fig_spectra.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
