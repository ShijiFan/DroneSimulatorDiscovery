"""W14: pre-generate rotor-return banks from fixed rotor_signal files with the W9 evaluator's drone prior.
Usage: python w14_build_banks.py <out_dir> ARM=path.py [ARM=path.py ...]
Each bank: x [N_BANK, 256] complex64 rotor-only return (no carrier, no beam, no body, no noise), plus the drawn params.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PRF, N = 17e3, 256
LAM = 299_792_458.0 / 77e9
N_BANK, SEED = 40000, 7


def load_fn(path):
    ns = {"np": np, "numpy": np, "math": math}
    exec(compile(Path(path).read_text(encoding="utf8"), str(path), "exec"), ns)
    return ns["rotor_signal"]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for spec in sys.argv[2:]:
        arm, path = spec.split("=", 1)
        fn = load_fn(path)
        rng = np.random.default_rng(SEED)
        t = np.arange(N) / PRF
        x = np.empty((N_BANK, N), np.complex64)
        prm = np.empty((N_BANK, 5))
        t0 = time.time()
        for i in range(N_BANK):
            L = rng.uniform(0.03, 0.25)
            f0 = rng.uniform(40, 150) / (2 * np.pi * L)
            n_b = int(rng.choice([2, 3]))
            n_rot = int(rng.choice([4, 6]))
            beta = np.deg2rad(rng.uniform(0, 30))
            r = np.asarray(fn(t.copy(), L, f0, n_b, n_rot, beta, LAM, rng), dtype=complex).ravel()
            if r.shape != (N,) or not np.all(np.isfinite(r)) or np.abs(r).max() == 0:
                raise ValueError(f"{arm}: bad output at {i}")
            x[i] = r / np.sqrt(np.mean(np.abs(r) ** 2))
            prm[i] = (L, f0, n_b, n_rot, beta)
        np.savez(out / f"bank_{arm}.npz", x=x, params=prm)
        meta[arm] = {"source": str(path), "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                     "bank_sha256": hashlib.sha256((out / f"bank_{arm}.npz").read_bytes()).hexdigest(),
                     "n": N_BANK, "seed": SEED, "gen_s": round(time.time() - t0, 1)}
        print(arm, meta[arm], flush=True)
    meta["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "banks.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
