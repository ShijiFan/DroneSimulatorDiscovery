import numpy as np

def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng):
    out = np.zeros(len(t), dtype=complex)
    k0 = 4 * np.pi / lam
    for k in range(n_rot):
        f = f0 * (1 + 0.05 * rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)
        for b in range(n_b):
            psi = 2 * np.pi * f * t + phi + 2 * np.pi * b / n_b
            x = k0 * L * np.cos(beta) * np.cos(psi)
            out += np.exp(0.5j * x) * np.sinc(x / (2 * np.pi))
        # hub / blade-root vibration: small radius, slow, dominant near DC
        rh = 0.01 * L
        xh = k0 * rh * np.cos(2*np.pi*f*t + phi)
        out += 0.12 * np.exp(1j * xh)
    # structural vibration of body-mounted parts (low-frequency, narrow-band)
    fv = rng.uniform(20, 300)
    av = rng.uniform(0.2, 1.0) * lam
    out += 0.3 * np.exp(1j * k0 * av * np.sin(2*np.pi*fv*t + rng.uniform(0, 6.28)))
    # airflow-driven parts (tail/prop wash) with net approaching Doppler
    fd = rng.uniform(0.5e3, 4.5e3)
    out += 0.35 * np.exp(-2j*np.pi*fd*t + 1j*rng.uniform(0, 6.28))
    return out
