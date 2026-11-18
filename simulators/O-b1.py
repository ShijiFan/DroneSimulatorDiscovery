import numpy as np

def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng):
    """Line blades + (a) always-visible hub features at very small radius (motor bell, hub cap edges)
    + (b) blade-root / pitch-link features at somewhat larger radius, shadowed by the hub housing
    on the approaching half of the rotation (one-sided visibility -> Doppler asymmetry)."""
    P_SYM = 2.2      # symmetric hub power / blade power
    P_ONE = 3.8      # one-sided root power / blade power
    OCC = 0.9
    n = len(t)
    out_b = np.zeros(n, dtype=complex)
    out_s = np.zeros(n, dtype=complex)
    out_o = np.zeros(n, dtype=complex)
    k0 = 4 * np.pi / lam * np.cos(beta)
    for k in range(n_rot):
        f = f0 * (1 + 0.05 * rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)
        for b in range(n_b):
            psi = 2 * np.pi * f * t + phi + 2 * np.pi * b / n_b
            c = np.cos(psi)
            x = k0 * L * c
            out_b += np.exp(0.5j * x) * np.sinc(x / (2 * np.pi))
            for r in 0.08 * rng.uniform(0, 1, 3) ** 3 * L:
                out_s += np.exp(1j * (k0 * r * c + rng.uniform(0, 2 * np.pi)))
            vis = 1 + OCC * np.sin(psi + 0.6)
            for r in (0.01 + 0.10 * rng.uniform(0, 1, 2) ** 2) * L:
                out_o += vis * np.exp(1j * (k0 * r * c + rng.uniform(0, 2 * np.pi)))
    pb = np.mean(np.abs(out_b) ** 2) + 1e-30
    ps = np.mean(np.abs(out_s) ** 2) + 1e-30
    po = np.mean(np.abs(out_o) ** 2) + 1e-30
    return out_b + out_s * np.sqrt(P_SYM * pb / ps) + out_o * np.sqrt(P_ONE * pb / po)
