"""W14 aggregate: apply the PROTOCOL W14 criteria to runs/W14_<arm>_s<seed>_*/result.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "runs"
ARMS, SEEDS = ["O", "S", "D", "V3"], [31, 32, 33]


def main():
    r = {}
    for a in ARMS:
        for s in SEEDS:
            hits = sorted(RUNS.glob(f"W14_{a}_s{s}_*/result.json"))
            if hits:
                r[(a, s)] = json.loads(hits[-1].read_text())
    tab = {}
    for a in ARMS:
        rs = [r[(a, s)] for s in SEEDS if (a, s) in r]
        if not rs:
            continue
        t = lambda f: [f(x) for x in rs]
        tab[a] = {"n_seeds": len(rs),
                  "open_auroc_mean": t(lambda x: x["track20"]["open_auroc_mean"]),
                  "gull": t(lambda x: x["track20"]["gull"]["auroc"]), "human": t(lambda x: x["track20"]["human"]["auroc"]),
                  "heron": t(lambda x: x["track20"]["heron"]["auroc"]), "CR": t(lambda x: x["track20"]["CR"]["auroc"]),
                  "closed_sim_auroc": t(lambda x: x["closed_sim_auroc"]),
                  "D2_D6_det_thr0": t(lambda x: float(np.mean([x["track20"]["detect_thr0"][c] for c in ["D2", "D3", "D4", "D5", "D6"]]))),
                  "D1_det_thr0": t(lambda x: x["track20"]["detect_thr0"]["D1"]),
                  "fa_thr0": {f: t(lambda x, f=f: x["track20"][f]["fa_thr0"]) for f in ["gull", "human", "heron", "CR"]},
                  "drone_median_logit": t(lambda x: x["track20"]["drone_median_logit"])}
    out = {"table": tab}
    if all((a, s) in r for a in ("O", "S") for s in SEEDS):
        o, s_ = np.array(tab["O"]["open_auroc_mean"]), np.array(tab["S"]["open_auroc_mean"])
        d = float(o.mean() - s_.mean())
        verdict = ("H1 supported" if d >= 0.02 and np.all(o > s_) else
                   "H1 reverse (fit-equivalent wrong physics equally useful)" if abs(d) < 0.01 else "inconclusive")
        out["H1"] = {"O_minus_S": d, "per_seed": (o - s_).tolist(), "verdict": verdict}
    if "D" in tab and tab["D"]["n_seeds"] == 3 and "O" in tab:
        out["secondary_D"] = {"D_minus_O": float(np.mean(tab["D"]["open_auroc_mean"]) - np.mean(tab["O"]["open_auroc_mean"])),
                              "pass": bool(np.mean(tab["D"]["open_auroc_mean"]) >= np.mean(tab["O"]["open_auroc_mean"]) - 0.01)}
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (RUNS / "W14_summary.json").write_text(json.dumps(out, indent=1))
    for a, v in tab.items():
        print(a, v["n_seeds"], "open %.4f" % np.mean(v["open_auroc_mean"]), "gull %.3f human %.3f heron %.3f" %
              (np.mean(v["gull"]), np.mean(v["human"]), np.mean(v["heron"])), "closed %.3f" % np.mean(v["closed_sim_auroc"]),
              "D2-6 %.2f D1 %.2f" % (np.mean(v["D2_D6_det_thr0"]), np.mean(v["D1_det_thr0"])),
              "FA heron %.2f gull %.2f human %.2f" % tuple(np.mean(v["fa_thr0"][f]) for f in ["heron", "gull", "human"]))
    print({k: out[k] for k in out if k in ("H1", "secondary_D")})


if __name__ == "__main__":
    main()
