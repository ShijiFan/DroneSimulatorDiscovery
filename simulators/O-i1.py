import numpy as np

def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng):
    """Line blades (Martin-Mulgrew) + motor bell / hub scatterers at small radii.
    Blade amplitude carries a pitch-induced aspect asymmetry: the blade face presented
    to the radar is larger on the receding half of the disc (negative Doppler)."""
    out = np.zeros(len(t), dtype=complex)
    k0 = 4 * np.pi / lam * np.cos(beta)
    a_asym = 0.47
    for k in range(n_rot):
        f = f0 * (1 + 0.05 * rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)
        for b in range(n_b):
            psi = 2 * np.pi * f * t + phi + 2 * np.pi * b / n_b
            x = k0 * L * np.cos(psi)
            amp = 4.5
            out += amp * np.exp(0.5j * x) * np.sinc(x / (2 * np.pi))
        # motor bell / shaft / prop nut: discrete scatterers at radii 3-15 mm
        n_s = 8
        r_hs = 0.002 + 0.012 * rng.uniform(0, 1, n_s) ** 2
        ph0 = rng.uniform(0, 2 * np.pi)
        for s in range(n_s):
            psi = 2 * np.pi * f * t + ph0 + 2 * np.pi * s / n_s
            a_s = a_asym * (0.030 - r_hs[s]) / 0.028
            amp = 0.5 * (1 + a_s * np.sin(psi))
            out += amp * np.exp(1j * k0 * r_hs[s] * np.cos(psi))
    return out
