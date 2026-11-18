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
