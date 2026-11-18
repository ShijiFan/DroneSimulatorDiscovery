import numpy as np


def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng):
    """Numpy port of the rotor part of sim_v2.py (hand-made v2): coherent line + diffuse blade points + hub points,
    leading-edge asymmetry exp(gamma sin(psi)/2); mixture weights drawn per target from PRIOR2."""
    a_spec = 10 ** (-17.9770 / 10)
    alpha = 2.8685
    gamma = 0.1310
    gamma_h = 0.7724
    hub_p = 10 ** (23.3038 / 10)
    r_hub = 0.00761
    cb = np.cos(beta)
    k = 4 * np.pi / lam
    spec = np.zeros(len(t), complex)
    diff = np.zeros(len(t), complex)
    hub = np.zeros(len(t), complex)
    for _ in range(n_rot):
        f = f0 * (1 + 0.05 * rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)
        for b in range(n_b):
            psi = 2 * np.pi * f * t + phi + 2 * np.pi * b / n_b
            asym = np.exp(0.5 * gamma * np.sin(psi))
            x = k * L * cb * np.cos(psi)
            spec += np.exp(0.5j * x) * np.sinc(x / (2 * np.pi)) * asym
            r = 0.15 * L + 0.85 * L * rng.uniform(size=8) ** (1 + alpha)
            amp = (rng.standard_normal(8) + 1j * rng.standard_normal(8)) / np.sqrt(2)
            diff += (amp[:, None] * np.exp(1j * k * cb * r[:, None] * np.cos(psi)[None, :])).sum(0) * asym
        ph = rng.uniform(0, 2 * np.pi, 4)
        psi_h = 2 * np.pi * f * t[None, :] + ph[:, None]
        amp_h = (rng.standard_normal(4) + 1j * rng.standard_normal(4)) / np.sqrt(2)
        hub += (amp_h[:, None] * np.exp(1j * k * r_hub * cb * np.cos(psi_h)) * np.exp(0.5 * gamma_h * np.sin(psi_h))).sum(0)
    pw = lambda z: max(np.mean(np.abs(z) ** 2), 1e-30)
    pd = pw(diff)
    return diff + spec * np.sqrt(pd / pw(spec) * a_spec) + hub * np.sqrt(pd / pw(hub) * hub_p)
