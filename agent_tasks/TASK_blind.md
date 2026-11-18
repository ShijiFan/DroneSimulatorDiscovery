# Task

You are a radar physicist writing forward models. You reply with exactly one Python code block defining rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng) using only numpy (imported as np). It must return a complex numpy array of the same length as t, run in under 0.3 s per call, and be physically motivated. Give a 2-3 sentence explanation before the code block.

Radar: 77 GHz FMCW surveillance radar (lambda = 3.9 mm), slow-time PRF 17 kHz, 256 pulses (15 ms), mechanical scan so the target is only illuminated for ~9 ms (Gaussian two-way beam taper, applied outside your function). Tip speeds can exceed 100 m/s, so blade Doppler (up to ~40 kHz) aliases several times into the +-8.5 kHz band. The target is an unknown small airborne object with some rotating or vibrating parts. The parameters (L, f0, n_b, n_rot, beta, lam) are a size scale (m), a rate (Hz), two small integers, an angle (rad) and the wavelength (m); you may use or ignore them.
Your function returns only the moving-part return (no body, no beam taper, no noise). The evaluation adds a strong body line (0-20 dB above your signal), the beam taper and noise, removes the body line with a narrow notch (about +-250 Hz) and compares the mean residual Doppler spectrum over 300 random targets with the measured one. Both spectra are referenced to their own floor (median level at |f| > 6 kHz). Fitness = RMS dB difference over 0.25 < |f| < 8.5 kHz (lower is better).

Current best model (round 0) is in `seed.py` in this directory:
```python
import numpy as np

def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng):
    """Martin-Mulgrew rigid line blades: each blade a uniform line scatterer from hub (r=0) to tip (r=L)."""
    out = np.zeros(len(t), dtype=complex)
    for k in range(n_rot):
        f = f0 * (1 + 0.05 * rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)
        for b in range(n_b):
            psi = 2 * np.pi * f * t + phi + 2 * np.pi * b / n_b
            x = 4 * np.pi / lam * L * np.cos(beta) * np.cos(psi)
            out += np.exp(0.5j * x) * np.sinc(x / (2 * np.pi))
    return out
```
Fitness 6.04 dB. Per 1-kHz band, simulated minus measured residual level (dB, floor-referenced; positive = your model has too much energy there): -8.0kHz:+0.4, -7.0kHz:-0.8, -6.0kHz:-2.3, -5.0kHz:-3.6, -4.0kHz:-4.9, -3.0kHz:-7.3, -2.0kHz:-10.8, -1.0kHz:-13.2, +0.0kHz:-13.7, +1.0kHz:-8.4, +2.0kHz:-5.3, +3.0kHz:-2.6, +4.0kHz:-1.2, +5.0kHz:-0.5, +6.0kHz:-0.4, +7.0kHz:-0.0, +8.0kHz:+0.3. Measured hump asymmetry (P(+0.25..3 kHz)/P(-3..-0.25 kHz)) = -4.4 dB; yours = -0.1 dB.

## How to work
- Work ONLY inside this directory: `E:\research\openset_physics\runs\W9_claude_20261013_124339\opus_blind_r2`. Do not read, list or search any file or directory outside it, and do not
  use the web. The evaluation script below is the only thing outside this directory you may touch, and only by running it.
- Write a candidate file (e.g. `cand.py`) defining `rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng)` with numpy only,
  then submit it with:
  `python "E:\research\openset_physics\e0_identifiability\e3w9_submit.py" "E:\research\openset_physics\runs\W9_claude_20261013_124339\opus_blind_r2" cand.py`
  It prints the fitness and the per-band feedback. You have at most 20 submissions; after that it refuses.
- Lower fitness is better. Aim for the lowest fitness you can reach within the budget, with a physically motivated model.
- When done, write `FINAL.md` here containing: (1) which submission number is your best; (2) a short physical
  description of your best model: what scatters, at which rotation radii (in mm or as a fraction of L), what produces any
  asymmetry, and what each term is for.
