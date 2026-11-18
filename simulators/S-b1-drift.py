import numpy as np

def rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng):
    out = np.zeros(len(t), dtype=complex)
    k0 = 4*np.pi/lam
    for k in range(n_rot):
        f = f0*(1+0.05*rng.uniform(-1,1))
        phi = rng.uniform(0,2*np.pi)
        for b in range(n_b):
            psi = 2*np.pi*f*t + phi + 2*np.pi*b/n_b
            x = k0*L*np.cos(beta)*np.cos(psi)
            out += np.exp(0.5j*x)*np.sinc(x/(2*np.pi))
    # hub / fuselage micro-vibration: small radius, slowly varying, plus drift (net radial velocity)
    a = 0.25*np.sqrt(n_rot)
    v = rng.uniform(-4.0, -0.3)
    fv = rng.uniform(80, 400)
    ph = rng.uniform(0, 2*np.pi)
    d = 0.003*rng.uniform(0.3,1.5)*np.sin(2*np.pi*fv*t+ph)
    out += a*np.exp(1j*k0*d)
    return out

