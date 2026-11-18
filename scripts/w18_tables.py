"""Paper tables after W18-W22: per-arm detection rates at logit 0 (track level, seed mean) from runs/W18_scores and W17 OOF
scores, merged with W18 CIs, W19 ablation, W20 LODO and fits.  Usage: python w18_tables.py  -> runs/W18_tables.json"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from w18_stats import DRONES, load_runs, tracks, w17_runs

RUNS = Path(__file__).resolve().parent.parent / "runs"
MAIN = ["TB", "K", "Km", "NB", "O2", "O", "O3", "O4", "D", "S", "S2", "S3", "S4"]
ABL = ["D", "D_noHub", "D_constVis", "D_noRoot", "D_bladeOnly"]


def pd_rates(lst):
    d1, d26 = [], []
    for seed, name, rec in lst:
        trk = {}
        for i, (l, s) in rec.items():
            trk.setdefault(l, []).append(tracks(s))
        d1.append(float((np.concatenate(trk["D1"]) > 0).mean()))
        d26.append(float(np.mean([(np.concatenate(trk[c]) > 0).mean() for c in DRONES[1:]])))
    return float(np.mean(d1)), float(np.mean(d26))


def fit_mean(paths):
    return float(np.mean([json.loads(Path(p).read_text())["fitness_rms_db"] for p in paths]))


def main():
    by = load_runs(set(MAIN + ABL))
    by.update(w17_runs())
    pd = {a: pd_rates(l) for a, l in by.items()}
    w18 = json.loads((RUNS / "W18_summary.json").read_text())
    w19 = json.loads((RUNS / "W18_summary_W19.json").read_text())
    fits = {f"D_{v}": fit_mean([RUNS / f"W19_ablation/fit_D_{v}_{s}.json" for s in (101, 202, 303)])
            for v in ("noHub", "noRoot", "constVis", "bladeOnly")}
    fits["Km"] = fit_mean([RUNS / f"W16_baselines/fit_Km_{s}.json" for s in (101, 202, 303)])
    out = {"pd": {a: {"D1": v[0], "D2_D6": v[1]} for a, v in pd.items()}, "fits_extra": fits,
           "main": {a: {"open": w18["point"][a]["open"], "ci": w18["ci95"][a]["open"], "heron": w18["point"][a]["heron"],
                        "tpr1": w18["point"][a]["tpr@0.01"], "CR": w18["point"][a]["CR"]} for a in w18["point"]},
           "ablation": {a: {"open": w19["point"][a]["open"], "ci": w19["ci95"][a]["open"], "heron": w19["point"][a]["heron"],
                            "tpr1": w19["point"][a]["tpr@0.01"]} for a in ABL},
           "diffs": w18["diffs"], "lodo": json.loads((RUNS / "W20_summary.json").read_text()),
           "lodo_shapes": json.loads((RUNS / "W20_lodo/shapes.json").read_text()),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (RUNS / "W18_tables.json").write_text(json.dumps(out, indent=1))
    for a in MAIN + ["realpos", "sup"]:
        print(a, {k: round(v, 3) for k, v in out["pd"][a].items()}, round(out["main"][a]["open"], 3))


if __name__ == "__main__":
    main()
