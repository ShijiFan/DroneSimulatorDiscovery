import numpy as np

def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng):
    """Blades as line scatterers with root-weighted reflectivity (thick root, thin tip), plus hub/mast
    vibration and a small net drift Doppler (descending/translating airframe -> asymmetry)."""
    out = np.zeros(len(t), dtype=complex)
    k = 4 * np.pi / lam
    for r in range(n_rot):
        f = f0 * (1 + 0.05 * rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)
        # rotor disc tilt: radial projection factor varies with azimuth
        tilt = rng.uniform(0, 0.3)
        # slow net Doppler from airframe motion/vibration, biased receding
        fd = rng.normal(-0.5e3, 0.5e3)
        amp_r = 10 ** (rng.normal(0, 3) / 20)
        sig = np.zeros(len(t), dtype=complex)
        for b in range(n_b):
            psi = 2 * np.pi * f * t + phi + 2 * np.pi * b / n_b
            x = k * L * np.cos(beta) * np.cos(psi)
            blade = np.exp(0.5j * x) * np.sinc(x / (2 * np.pi))
            # root segment (r ~0.2L) with thicker chord: lower Doppler, stronger
            xr = 0.12 * x
            root = 1.5 * np.exp(1j * xr)
            # tip hot spot (r = L) leading-edge flash
            tip = 0.4 * np.exp(1j * x)
            sig += 0.6*blade + 4.0*root + 0.1*tip
        sig *= amp_r * np.exp(2j * np.pi * fd * t)
        # hub vibration
        fv = rng.normal(-0.9e3, 0.9e3); sig += 16.0 * np.exp(1j * 2.0 * np.sin(2 * np.pi * f * n_b * t + phi)) * np.exp(2j * np.pi * fv * t)
        out += sig
    return out
