"""W9-3 (PROTOCOL.md): fit a parametric family written by a distiller sub-agent with the same optimiser as v3
(120 random-search + 180 Nelder-Mead evaluations at seed 2026), then re-evaluate the best on fresh seeds 101/202/303.
The family file must define PARAMS = [(name, lo, hi), ...] (<= 6 entries), a single-line THETA = {...} assignment,
and rotor_signal(t, L, f0, n_b, n_rot, beta, lam, rng) that reads its parameters from THETA.
Usage: python e3w9_fit_family.py <family.py> <real_shape.npz> <out dir>
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent


def main():
    fam, real, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    src = fam.read_text(encoding="utf8")
    tree = ast.parse(src)
    params = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PARAMS")
    assert 1 <= len(params) <= 6, "PARAMS must have 1-6 entries"
    names = [p[0] for p in params]
    lo, hi = np.array([p[1] for p in params], float), np.array([p[2] for p in params], float)
    pat = re.compile(r"^THETA\s*=.*$", re.M)
    assert len(pat.findall(src)) == 1, "need exactly one single-line THETA = {...}"

    def code_for(th):
        th = np.clip(th, lo, hi)
        return pat.sub("THETA = " + repr(dict(zip(names, map(float, th)))), src)

    def run_eval(code, seed):
        p = out / "cand.py"
        p.write_text(code, encoding="utf8")
        r = subprocess.run([sys.executable, "-I", str(HERE / "e3w7_eval.py"), str(p), str(real), str(out / "cand.json"),
                            str(seed)], capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return None
        return json.loads((out / "cand.json").read_text())

    hist = []

    def f(th):
        res = run_eval(code_for(th), 2026)
        fit = 99.0 if res is None else res["fitness_rms_db"]
        hist.append({"theta": dict(zip(names, map(float, np.clip(th, lo, hi)))), "fitness": fit})
        return fit

    rng = np.random.default_rng(0)
    for _ in range(120):
        f(lo + (hi - lo) * rng.uniform(size=len(names)))
    x0 = np.array(list(min(hist, key=lambda h: h["fitness"])["theta"].values()))
    minimize(f, x0, method="Nelder-Mead", options={"maxfev": 180})
    best = min(hist, key=lambda h: h["fitness"])
    code = code_for(np.array(list(best["theta"].values())))
    (out / "best.py").write_text(code, encoding="utf8")
    fresh = {s: run_eval(code, s) for s in (101, 202, 303)}
    res = {"family": str(fam), "family_sha256": hashlib.sha256(fam.read_bytes()).hexdigest(), "n_params": len(names),
           "n_evals": len(hist), "n_failed": sum(h["fitness"] == 99.0 for h in hist),
           "best_seed2026": best["fitness"], "theta": best["theta"],
           "fresh_fitness": {s: v["fitness_rms_db"] for s, v in fresh.items()},
           "fresh_mean": float(np.mean([v["fitness_rms_db"] for v in fresh.values()])),
           "fresh_asym": {s: v["sim_asym_db_hump"] for s, v in fresh.items()},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    (out / "history.json").write_text(json.dumps(hist))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
