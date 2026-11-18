"""W20 (PROTOCOL W18-W21): leave-one-drone-type-out calibration.  For each held-out type k, compare detectors trained on
the family refitted without k (arm L_wo_k) against the full-calibration family (arm D, W14) on the same subset:
positives = tracks of type k only, negatives = gull / human / heron tracks.  H20: mean over k of (LODO - full) >= -0.02.
Usage: python w20_eval.py   (needs runs/W18_scores for W14_D_* and W14_L_wo_*)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from w18_stats import FAMS, OPEN, auc, load_runs, tracks

RUNS = Path(__file__).resolve().parent.parent / "runs"
TYPES = ["D1", "D2", "D3", "D4", "D5", "D6"]


def held_out_open(rec, k):
    trk = {}
    for i, (l, s) in rec.items():
        trk.setdefault(l, []).append(tracks(s))
    pos = np.concatenate(trk[k])
    return float(np.mean([auc(pos, np.concatenate([t for c in FAMS[f] for t in trk.get(c, [])])) for f in OPEN]))


def main():
    by = load_runs({"D"} | {f"L_wo_{k}" for k in TYPES})
    out = {"folds": {}}
    for k in TYPES:
        lodo = [held_out_open(rec, k) for s, n, rec in by[f"L_wo_{k}"]]
        full = [held_out_open(rec, k) for s, n, rec in by["D"]]
        fit = json.loads((RUNS / f"W20_lodo/fit_wo_{k}/result.json").read_text())
        out["folds"][k] = {"lodo_open": float(np.mean(lodo)), "lodo_seeds": lodo, "full_open": float(np.mean(full)),
                           "full_seeds": full, "diff": float(np.mean(lodo) - np.mean(full)),
                           "theta_refit": fit["theta"], "fresh_fit_db_own_summary": fit["fresh_mean"]}
        print(f"{k}: LODO {np.mean(lodo):.3f}  full {np.mean(full):.3f}  diff {np.mean(lodo) - np.mean(full):+.3f}")
    d = [v["diff"] for v in out["folds"].values()]
    out["mean_diff"] = float(np.mean(d))
    out["H20_pass"] = bool(np.mean(d) >= -0.02)
    out["mean_lodo_open"] = float(np.mean([v["lodo_open"] for v in out["folds"].values()]))
    out["mean_full_open"] = float(np.mean([v["full_open"] for v in out["folds"].values()]))
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (RUNS / "W20_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("mean_diff", "H20_pass", "mean_lodo_open", "mean_full_open")}, indent=1))


if __name__ == "__main__":
    main()
