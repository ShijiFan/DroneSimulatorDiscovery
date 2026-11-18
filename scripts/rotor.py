"""Single-rotor micro-Doppler forward model, classical estimators, NLS refinement and CRB.

Geometry follows the Martin-Mulgrew rotating-blade model: each blade is a uniform line
scatterer from the hub (r=0) to the tip (r=L), integrated in closed form. The rotor plane
is tilted by a known aspect angle beta from the radar line of sight, so only L*cos(beta)
enters the phase; beta is treated as known (radar elevation of a level multirotor).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

C = 299_792_458.0


def blade_sum(t, wavelength, L, f_rot, phi0, n_blades, beta):
    """Closed-form complex return of all blades (unit reflectivity per metre)."""
    a = 4.0 * np.pi / wavelength * np.cos(beta)
    out = np.zeros_like(t, dtype=np.complex128)
    for k in range(n_blades):
        psi = 2.0 * np.pi * f_rot * t + phi0 + 2.0 * np.pi * k / n_blades
        x = a * L * np.cos(psi)
        # integral_0^L exp(j a r cos psi) dr = L exp(j x/2) sinc(x / (2 pi))
        out += L * np.exp(0.5j * x) * np.sinc(x / (2.0 * np.pi))
    return out


def simulate(rng, fc, prf, dwell, L, f_rot, phi0, n_blades, beta, snr_db, body_to_rotor_db):
    """Slow-time IQ: rotor + static body + complex white noise. SNR is rotor power / noise power."""
    wavelength = C / fc
    t = np.arange(int(round(prf * dwell))) / prf
    rotor = blade_sum(t, wavelength, L, f_rot, phi0, n_blades, beta)
    p_rotor = np.mean(np.abs(rotor) ** 2)
    body = np.sqrt(p_rotor * 10 ** (body_to_rotor_db / 10)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    sigma2 = p_rotor / 10 ** (snr_db / 10)
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))
    return t, rotor + body + noise, sigma2


# ---------------------------------------------------------------- classical estimators

def estimate_line_spacing(x, prf, n_blades, f_rot_bounds=(25.0, 200.0), n_harm=60):
    """The rotor return is periodic with period 1/(N_b f_rot) for any blade count, so the
    slow-time spectrum is a comb with spacing N_b f_rot. Harmonic-comb score with half-step
    penalty to suppress octave errors. Works under Doppler aliasing (the comb folds onto itself)."""
    n = x.size
    nfft = int(2 ** np.ceil(np.log2(n) + 3))
    spec = np.abs(np.fft.fft(x - x.mean(), nfft)) ** 2
    spec /= np.median(spec) + 1e-30
    freqs = np.fft.fftfreq(nfft, 1 / prf)
    lo, hi = n_blades * f_rot_bounds[0], n_blades * f_rot_bounds[1]
    cands = np.arange(lo, hi, 0.05 * prf / n)

    def comb(delta, offsets):
        k = np.arange(1, n_harm + 1)
        f = np.concatenate([(k + o) * delta for o in offsets] + [-(k + o) * delta for o in offsets])
        idx = np.round((np.mod(f + prf / 2, prf) - prf / 2) / (prf / nfft)).astype(int) % nfft
        return np.log1p(spec[idx]).mean()

    def score(d):
        # penalise candidates that are 2x or 3x the true spacing: real lines then fall at 1/2, 1/3, 2/3 offsets
        return comb(d, [0.0]) - max(comb(d, [0.5]), comb(d, [1 / 3]), comb(d, [2 / 3]))

    scores = np.array([score(d) for d in cands])
    best = cands[int(np.argmax(scores))]
    fine = np.linspace(best - prf / n, best + prf / n, 81)
    fs = np.array([score(d) for d in fine])
    return fine[int(np.argmax(fs))]


def estimate_max_doppler(x, prf, thresh_db=6.0):
    """Doppler extent of the blade spectrum above the noise floor (fails under aliasing)."""
    n = x.size
    seg = min(256, n)
    hop = seg // 2
    win = np.hanning(seg)
    frames = [x[i:i + seg] * win for i in range(0, n - seg + 1, hop)]
    p = np.mean(np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2, axis=0)
    f = np.fft.fftshift(np.fft.fftfreq(seg, 1 / prf))
    floor = np.median(p)
    above = np.abs(f)[(p > floor * 10 ** (thresh_db / 10)) & (np.abs(f) > 2 * prf / seg)]
    return float(above.max()) if above.size else np.nan


def estimate_period_autocorr(x, prf, n_blades, f_rot_bounds=(25.0, 200.0), rel=0.85):
    """Period 1/(N_b f_rot) from the unbiased slow-time autocorrelation (full band, alias-safe).
    The smallest lag whose peak reaches `rel` of the largest peak is taken, to avoid 2x/3x periods."""
    x0 = x - x.mean()
    n = x0.size
    spec = np.fft.fft(x0, 2 * n)
    r = np.abs(np.fft.ifft(np.abs(spec) ** 2)[:n]) / (n - np.arange(n))
    lo = int(np.floor(prf / (n_blades * f_rot_bounds[1])))
    hi = min(int(np.ceil(prf / (n_blades * f_rot_bounds[0]))), int(0.8 * n))
    if hi <= lo + 2:
        return np.nan
    seg = r[lo:hi]
    peaks = [i for i in range(1, seg.size - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return np.nan
    vmax = max(seg[i] for i in peaks)
    i = min(p for p in peaks if seg[p] >= rel * vmax)
    y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return (lo + i + frac) / prf


def classical(x, prf, fc, n_blades, beta):
    wavelength = C / fc
    period = estimate_period_autocorr(x, prf, n_blades)
    f_rot = 1.0 / (n_blades * period) if np.isfinite(period) else np.nan
    fmax = estimate_max_doppler(x, prf)
    v_tip = fmax * wavelength / (2 * np.cos(beta))
    return {"f_rot": f_rot, "v_tip": v_tip, "L": v_tip / (2 * np.pi * f_rot)}


# ---------------------------------------------------------------- NLS (variable projection)

def residual_energy(theta, t, x, wavelength, n_blades, beta):
    L, f_rot, phi0 = theta
    if L <= 0 or f_rot <= 0:
        return np.inf
    A = np.stack([blade_sum(t, wavelength, L, f_rot, phi0, n_blades, beta), np.ones_like(t, dtype=complex)], 1)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ coef
    return float(np.vdot(r, r).real)


def nls(t, x, fc, n_blades, beta, init, n_phase=24, df_span=3, df_steps=7):
    """Local NLS from a classical initialisation: grid over (f_rot near init, phi0), then Nelder-Mead."""
    wavelength = C / fc
    T = t[-1] - t[0]
    best = (np.inf, None)
    f_grid = init["f_rot"] + np.linspace(-df_span, df_span, df_steps) * (0.02 / max(T, 1e-3))
    L0 = init["L"] if np.isfinite(init["L"]) else 0.15
    for L_try in (L0, 0.9 * L0, 1.1 * L0):
        for f in f_grid:
            for ph in np.linspace(0, 2 * np.pi / n_blades, n_phase, endpoint=False):
                e = residual_energy((L_try, f, ph), t, x, wavelength, n_blades, beta)
                if e < best[0]:
                    best = (e, (L_try, f, ph))
    res = minimize(
        residual_energy, np.array(best[1]), args=(t, x, wavelength, n_blades, beta),
        method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 2000},
    )
    L, f_rot, phi0 = res.x
    return {"L": float(L), "f_rot": float(f_rot), "phi0": float(phi0)}


# ---------------------------------------------------------------- CRB

def crb(t, fc, L, f_rot, phi0, n_blades, beta, sigma2, a_rotor=1.0, b_body=1.0):
    """CRB for (L, f_rot) with phi0 and the two complex amplitudes as nuisance parameters."""
    wavelength = C / fc

    def s(th):
        return blade_sum(t, wavelength, th[0], th[1], th[2], n_blades, beta)

    th = np.array([L, f_rot, phi0])
    steps = np.array([1e-6, 1e-6, 1e-6]) * np.maximum(np.abs(th), 1.0)
    cols = []
    for i in range(3):
        d = np.zeros(3)
        d[i] = steps[i]
        cols.append(a_rotor * (s(th + d) - s(th - d)) / (2 * steps[i]))
    s0 = s(th)
    one = np.ones_like(t, dtype=complex)
    cols += [s0, 1j * s0, one, 1j * one]
    J = np.stack(cols, 1)
    fim = 2.0 / sigma2 * np.real(J.conj().T @ J)
    cov = np.linalg.pinv(fim)
    return {"L": float(np.sqrt(cov[0, 0])), "f_rot": float(np.sqrt(cov[1, 1]))}
