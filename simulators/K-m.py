import numpy as np


def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng):
    """W22 prior-matched variant of K_karlsson2023.py (Karlsson PhD thesis, KTH 2023, eqs. (6.4)-(6.8)).
    Scattering mechanism unchanged (effective blade length d ~ U(1, 20) mm, weights alpha1 ~ U(0,1), per-rotor rate jitter
    delta ~ U(0.5, 1.5)); the rotation rate nu and the number of rotors come from the caller (shared drone prior) instead of
    the thesis prior (nu ~ U(10, 200) rev/s, 1-10 propellers)."""
    nu = f0                                   # caller's rotation rate (rev/s)
    d = rng.uniform(0.001, 0.02)              # effective blade length [m], thesis Table 6.4
    n_p = int(n_rot)                          # caller's number of rotors
    a1 = rng.uniform(0, 1)
    alpha = (a1, 1 - a1)
    ns = int(np.ceil(10 * d / lam))
    di = d * np.arange(1, ns + 1) / ns
    out = np.zeros(len(t), dtype=complex)
    for _ in range(n_p):
        delta = rng.uniform(0.5, 1.5)
        phi = rng.uniform(0, 2 * np.pi)
        s = np.sin(2 * np.pi * delta * nu * t + phi)
        th = 2 * np.pi * di[:, None] * s[None, :] / lam      # [Ns, T]
        ind = np.where(np.cos(th) >= 0, 1.0, -1.0)
        sinc = np.sinc(th / np.pi)
        for k, a in zip((1, 2), alpha):
            out += a * (np.exp(1j * (-1) ** k * ind * th) * sinc).sum(0)
    return out
