"""E3w W7 evaluator (self-contained, run with python -I in a subprocess).
Loads a candidate file defining rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng) -> complex array (len t), wraps it in
the fixed SAAB-like chain (body 0-20 dB, Gaussian beam FWHM 9 ms, SNR 20 dB re rotor), computes the mean Legendre-notch
residual spectrum and compares its floor-referenced shape with the real D1-D6 pooled shape.
Usage: python -I e3w7_eval.py <candidate.py> <real_shape.npz> <out.json> [seed, default 2026]
"""

import json
import math
import sys
import time

import numpy as np

C, FC, PRF, N, NF, FWHM = 299_792_458.0, 77e9, 17e3, 256, 512, 9e-3
LAM = C / FC
N_DRONES = 300
T = np.linspace(-1, 1, N)
LEG = np.stack([np.polynomial.legendre.Legendre.basis(k)(T) for k in range(5)], 1)


def spec(x):
    f = np.fft.fft(x * np.hanning(N), 8 * N)
    fd = np.argmax(np.abs(f)) / (8 * N)
    y = x * np.exp(-2j * np.pi * fd * np.arange(N))
    coef, *_ = np.linalg.lstsq(LEG, y, rcond=None)
    b = LEG @ coef
    r = y - b
    w = np.hanning(N)
    R = np.abs(np.fft.fftshift(np.fft.fft(r * w, NF))) ** 2
    return R / (np.abs(b * w) ** 2).sum()


def shape(P, f):
    db = 10 * np.log10(P + 1e-30)
    return db - np.median(db[np.abs(f) > 6e3])


def main():
    cand, real_npz, out = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    ns = {"np": np, "numpy": np, "math": math}
    src = open(cand, encoding="utf8").read()
    exec(compile(src, cand, "exec"), ns)
    fn = ns["rotor_signal"]
    rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 2026)
    t = np.arange(N) / PRF
    specs = []
    for _ in range(N_DRONES):
        L = rng.uniform(0.03, 0.25)
        v_tip = rng.uniform(40, 150)
        f0 = v_tip / (2 * np.pi * L)
        n_b = int(rng.choice([2, 3]))
        n_rot = int(rng.choice([4, 6]))
        beta = np.deg2rad(rng.uniform(0, 30))
        rot = np.asarray(fn(t.copy(), L, f0, n_b, n_rot, beta, LAM, rng), dtype=complex).ravel()
        if rot.shape != (N,) or not np.all(np.isfinite(rot)) or np.abs(rot).max() == 0:
            raise ValueError(f"rotor_signal must return {N} finite, non-zero complex samples; got shape {rot.shape}")
        v_r = rng.uniform(-5, 5)
        carrier = np.exp(2j * np.pi * (2 * v_r / LAM) * t)
        tc = 7.5e-3 + rng.uniform(-1e-3, 1e-3)
        w = np.exp(-4 * np.log(2) * (t - tc) ** 2 / FWHM ** 2)
        rot = rot * carrier * w
        p = np.mean(np.abs(rot) ** 2)
        body = np.sqrt(p * 10 ** (rng.uniform(0, 20) / 10)) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * carrier * w
        noise = np.sqrt(p / 100 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        specs.append(spec(rot + body + noise))
    f = np.fft.fftshift(np.fft.fftfreq(NF, 1 / PRF))
    sim = shape(np.mean(specs, 0), f)
    real = np.load(real_npz)["shape"]
    m = (np.abs(f) > 250) & (np.abs(f) < 8.5e3)
    fit = float(np.sqrt(np.mean((sim[m] - real[m]) ** 2)))
    edges = np.arange(-8.5e3, 8.6e3, 1e3)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi) & (np.abs(f) > 250)
        bands.append(round(float(np.mean(sim[sel] - real[sel])), 1) if sel.any() else None)
    hump = (np.abs(f) > 250) & (np.abs(f) < 3e3)
    P = np.mean(specs, 0)
    asym = float(10 * np.log10(P[hump & (f > 0)].sum() / P[hump & (f < 0)].sum()))
    json.dump({"fitness_rms_db": fit, "band_centres_khz": [float((a + b) / 2e3) for a, b in zip(edges[:-1], edges[1:])],
               "sim_minus_real_db": bands, "sim_asym_db_hump": asym, "runtime_s": round(time.time() - t0, 1)},
              open(out, "w"))


if __name__ == "__main__":
    main()
