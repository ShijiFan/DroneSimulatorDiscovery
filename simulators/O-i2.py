import numpy as np

def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng, A_DIF=0.3, P_DIF=4.0, A_HUB=1.7, R_HUB=(0.0008, 0.0095), A_ASYM=0.5, A_SPEC=0.4, N_HUB=3, A_ASYM_D=-0.3):
    """Specular line blades + diffuse point scatterers along blade (root-weighted) + motor-bell scatterers.
    Aspect-dependent amplitude (1 + a*sin psi) gives approaching/receding asymmetry."""
    t = np.asarray(t, dtype=float)
    out = np.zeros(len(t), dtype=complex)
    k = 4 * np.pi / lam * np.cos(beta)
    n_pts = 12
    for kr in range(n_rot):
        f = f0 * (1 + 0.05 * rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)
        for b in range(n_b):
            psi = 2 * np.pi * f * t + phi + 2 * np.pi * b / n_b
            c = np.cos(psi); s = np.sin(psi)
            x = k * L * c
            out += A_SPEC * np.exp(0.5j * x) * np.sinc(x / (2 * np.pi))
            r = L * rng.uniform(0.05, 1.0, n_pts) ** P_DIF
            amp = A_DIF * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts)) / np.sqrt(2)
            w = 1 + A_ASYM_D * s
            out += w * (amp[None, :] * np.exp(1j * k * r[None, :] * c[:, None])).sum(1)
        for m in range(N_HUB):
            rm = rng.uniform(*R_HUB)
            psi = 2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)
            out += A_HUB * (1 + A_ASYM * np.sin(psi)) * np.exp(1j * k * rm * np.cos(psi)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return out
