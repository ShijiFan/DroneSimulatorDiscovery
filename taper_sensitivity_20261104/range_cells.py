"""André RS review check: residual Doppler power per range cell (5 cells kept per dwell; cell 2 = analysed cell). If sawtooth
range-Doppler coupling pushed blade-tip (high-Doppler) energy into neighbouring cells, the high/low band ratio would rise there."""
import sys, numpy as np
sys.path.insert(0, r'E:\research\openset_physics\e0_identifiability')
import e3w7_eval as E
d = np.load(r'E:\rag\datasets\data3\data_SAAB_SIRS_77GHz_FMCW.npy', allow_pickle=True)
lab = np.array([r[0][0] for r in d]); rng = np.random.default_rng(0)
f = np.fft.fftshift(np.fft.fftfreq(E.NF, 1 / E.PRF)); lo = (np.abs(f) > 250) & (np.abs(f) < 3e3); hi = (np.abs(f) >= 3e3) & (np.abs(f) < 8.5e3)
w = np.hanning(E.N)
def resid_pow(x):
    F = np.fft.fft(x * w, 8 * E.N); fd = np.argmax(np.abs(F)) / (8 * E.N); y = x * np.exp(-2j * np.pi * fd * np.arange(E.N))
    c, *_ = np.linalg.lstsq(E.LEG, y, rcond=None); r = y - E.LEG @ c
    return np.abs(np.fft.fftshift(np.fft.fft(r * w, E.NF))) ** 2
for cls in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'seagull', 'human_walk']:
    P = np.zeros((5, E.NF)); tot = np.zeros(5); n = 0
    for i in np.flatnonzero(lab == cls):
        X = d[i, 1]; cols = rng.choice(X.shape[1], min(60, X.shape[1]), replace=False)
        for j in cols:
            for c in range(5):
                x = X[c * 256:(c + 1) * 256, j]; P[c] += resid_pow(x); tot[c] += np.sum(np.abs(x) ** 2)
            n += 1
    ref_lo = P[2][lo].sum(); ref_hi = P[2][hi].sum()
    print(f'{cls:11s} n={n:4d} | total power dB rel cell2: ' + ' '.join(f'{10*np.log10(tot[c]/tot[2]):6.1f}' for c in range(5)) +
          ' | lo-band dB: ' + ' '.join(f'{10*np.log10(P[c][lo].sum()/ref_lo):6.1f}' for c in range(5)) +
          ' | hi-band dB: ' + ' '.join(f'{10*np.log10(P[c][hi].sum()/ref_hi):6.1f}' for c in range(5)) +
          ' | hi/lo dB: ' + ' '.join(f'{10*np.log10(P[c][hi].sum()/P[c][lo].sum()):6.1f}' for c in range(5)))
