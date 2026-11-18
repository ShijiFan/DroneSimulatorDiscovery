"""André RS review check (2026-11-04): spectral misfit of the released simulators with the measured beam taper
(amplitude FWHM ~5 ms) vs the paper's 9 ms. Same chain as e3w7_eval.py, three fresh seeds."""
import sys, json, math, glob, os
import numpy as np
sys.path.insert(0, r'E:\research\openset_physics\e0_identifiability')
import e3w7_eval as E
SIM = r'E:\research\openset_physics\paper_RS\FINAL_SUBMISSION_RS_20261015\04_Code_and_Programs\simulators'
real = np.load(r'E:\research\openset_physics\runs\E3w7_20261013_004649\blind\real_shape.npz')['shape']
f = np.fft.fftshift(np.fft.fftfreq(E.NF, 1 / E.PRF)); m = (np.abs(f) > 250) & (np.abs(f) < 8.5e3)
def fit(fn, fwhm, seed):
    rng = np.random.default_rng(seed); t = np.arange(E.N) / E.PRF; specs = []
    for _ in range(E.N_DRONES):
        L = rng.uniform(0.03, 0.25); v_tip = rng.uniform(40, 150); f0 = v_tip / (2 * np.pi * L)
        n_b = int(rng.choice([2, 3])); n_rot = int(rng.choice([4, 6])); beta = np.deg2rad(rng.uniform(0, 30))
        rot = np.asarray(fn(t.copy(), L, f0, n_b, n_rot, beta, E.LAM, rng), complex).ravel()
        v_r = rng.uniform(-5, 5); car = np.exp(2j * np.pi * (2 * v_r / E.LAM) * t); tc = 7.5e-3 + rng.uniform(-1e-3, 1e-3)
        w = np.exp(-4 * np.log(2) * (t - tc) ** 2 / fwhm ** 2); rot = rot * car * w; p = np.mean(np.abs(rot) ** 2)
        body = np.sqrt(p * 10 ** (rng.uniform(0, 20) / 10)) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * car * w
        noise = np.sqrt(p / 100 / 2) * (rng.standard_normal(E.N) + 1j * rng.standard_normal(E.N))
        specs.append(E.spec(rot + body + noise))
    sim = E.shape(np.mean(specs, 0), f); return float(np.sqrt(np.mean((sim[m] - real[m]) ** 2)))
out = {}
for p in sorted(glob.glob(SIM + r'\*.py')):
    name = os.path.basename(p)[:-3]
    if 'template' in name: continue
    ns = {'np': np, 'numpy': np, 'math': math}; exec(compile(open(p, encoding='utf8').read(), p, 'exec'), ns)
    if 'rotor_signal' not in ns: continue
    out[name] = {str(fw): round(float(np.mean([fit(ns['rotor_signal'], fw, s) for s in (101, 202, 303)])), 3) for fw in (9e-3, 5e-3)}
    print(name, out[name], flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
