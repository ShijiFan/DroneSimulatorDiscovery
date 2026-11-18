import numpy as np

def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng):
    out = np.zeros(len(t), dtype=complex)
    k0 = 4 * np.pi / lam * np.cos(beta)
    for k in range(n_rot):
        f = f0 * (1 + 0.05 * rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)
        for b in range(n_b):
            psi = 2 * np.pi * f * t + phi + 2 * np.pi * b / n_b
            c = np.cos(psi)
            x = k0 * L * c
            out += 0.7 * np.exp(0.5j * x) * np.sinc(x / (2 * np.pi))
            out += 0.03 * np.exp(1j * k0 * 0.25 * L * c) * np.exp(-(np.sin(psi) / 0.15) ** 2)
        out += 0.15 * np.exp(1j * k0 * 0.05 * L * np.cos(2 * np.pi * f * t + phi))
    out *= np.exp(1j * (-2 * np.pi * 800 * t + 0.7 * np.sin(2 * np.pi * rng.uniform(20, 60) * t + rng.uniform(0, 6.3))))
    return out




