import numpy as np


def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng):
    """Karlsson PhD thesis (KTH 2023) ch. 6, eqs. (6.4)-(6.8), parameters drawn from Table 6.4 (anomaly approach).
    Ignores the caller's L, f0, n_b, n_rot, beta: the expert model uses its own prior, as in the thesis.
    z = sum_n sum_i sum_k alpha_k exp(j (-1)^k I(theta) theta) sinc(theta),
    theta_{n,i} = 2 pi d_i sin(2 pi delta_n nu t + phi_n) / lam, d_i = i d / Ns, Ns = ceil(10 d / lam),
    I(theta) = +1 if cos(theta) >= 0 else -1, sinc non-normalised.  Reproduced literally."""
    nu = rng.uniform(10, 200)                 # RPS
    d = rng.uniform(0.001, 0.02)              # effective blade length [m]
    n_p = int(rng.integers(1, 11))            # number of propellers, U{1..10}
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
