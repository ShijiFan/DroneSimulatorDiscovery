"""W18 step 2 (PROTOCOL W18): recording-clustered bootstrap CIs, TPR at fixed FPR, corner reflector / pigeon / raven.
Reads runs/W18_scores/*.npz (from w18_rescore.py) and W17 out-of-fold scores.  Usage: python w18_stats.py [arm ...]
Writes runs/W18_summary.json (or W18_summary_<tag>.json when arms are given via --tag)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

RUNS = Path(__file__).resolve().parent.parent / "runs"
DRONES = ["D1", "D2", "D3", "D4", "D5", "D6"]
FAMS = {"gull": ["seagull", "black-headed gull", "seagull and black-headed gull"], "human": ["human_walk", "human_run"],
        "heron": ["heron"], "CR": ["CR"]}
OPEN = ["gull", "human", "heron"]
ARMS = ["TB", "K", "Km", "NB", "O2", "O", "O3", "O4", "D", "S", "S2", "S3", "S4", "E1", "E2", "E3", "E4"]
OPUS = ["O2", "O", "O3", "O4", "D"]
TRACK, B = 20, 2000


def auc(pos, neg):
    """Mann-Whitney AUROC with ties counted 1/2."""
    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def tracks(dw):
    n = len(dw) // TRACK
    return dw[:n * TRACK].reshape(n, TRACK).mean(1) if n else np.empty(0)


def load_runs(arms):
    by = {}
    for p in sorted((RUNS / "W18_scores").glob("*.npz")):
        m = re.match(r"(?:W14|W19|W20)_(.+)_s(\d+)_\d{8}_\d{6}$", p.stem)
        if not m or m[1] not in arms:
            continue
        z = np.load(p)
        rec = {int(i): (str(l), z[f"r{int(i):03d}"]) for l, i in zip(z["labels"], z["index"])}
        by.setdefault(m[1], []).append((int(m[2]), p.stem, rec))
    return by


def w17_runs():
    out = {}
    for mode in ("realpos", "sup"):
        rec = {}
        for p in sorted(RUNS.glob(f"W17_{mode}_f*_*/oof_scores.json")):
            meta = json.loads((p.parent / "meta.json").read_text())
            seed = meta["args"]["seed"]
            for key, s in json.loads(p.read_text()).items():
                i, lab = key.split("|", 1)
                rec.setdefault(seed, {})[int(i)] = (lab, np.asarray(s, float))
        out[mode] = [(seed, f"W17_{mode}_s{seed}", r) for seed, r in sorted(rec.items()) if len(r) > 100]
    return out


def stats_for(rec_tracks, groups):
    """rec_tracks: {rec: track scores}; groups: {class: [rec ids]} (possibly resampled)."""
    pos = np.concatenate([rec_tracks[r] for c in DRONES for r in groups[c]])
    out = {}
    for fam, cls in FAMS.items():
        neg = np.concatenate([rec_tracks[r] for c in cls for r in groups.get(c, [])])
        out[fam] = auc(pos, neg) if len(neg) else np.nan
    out["open"] = float(np.mean([out[f] for f in OPEN]))
    negall = np.concatenate([rec_tracks[r] for f in OPEN for c in FAMS[f] for r in groups.get(c, [])])
    for fpr in (0.01, 0.05, 0.10):
        out[f"tpr@{fpr:.2f}"] = float((pos > np.quantile(negall, 1 - fpr)).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", default=ARMS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    by = load_runs(set(args.arms))
    by.update(w17_runs())
    # sanity: recomputed track AUROC equals the stored result for W14-chain runs
    worst = 0.0
    for arm, lst in by.items():
        for seed, name, rec in lst:
            rj = RUNS / name / "result.json"
            if rj.exists():
                r = json.loads(rj.read_text())["track20"]
                trk = {i: tracks(s) for i, (l, s) in rec.items()}
                g = {}
                for i, (l, s) in rec.items():
                    g.setdefault(l, []).append(i)
                worst = max(worst, abs(stats_for(trk, g)["open"] - r["open_auroc_mean"]))
    assert worst < 1e-6, worst
    any_rec = next(iter(by.values()))[0][2]
    labels = {i: l for i, (l, s) in any_rec.items()}
    groups = {}
    for i, l in labels.items():
        groups.setdefault(l, []).append(i)
    classes = [c for c in groups if c in DRONES or any(c in v for v in FAMS.values())]
    trk = {arm: [(seed, {i: tracks(s) for i, (l, s) in rec.items()}) for seed, name, rec in lst] for arm, lst in by.items()}
    point = {arm: {k: float(np.mean([stats_for(t, groups)[k] for seed, t in lst])) for k in stats_for(lst[0][1], groups)}
             for arm, lst in trk.items()}
    rng = np.random.default_rng(0)
    boot = {arm: [] for arm in trk}
    for b in range(B):
        g = {c: list(rng.choice(groups[c], len(groups[c]), replace=True)) for c in classes}
        for arm, lst in trk.items():
            vals = [stats_for(t, g) for seed, t in lst]
            boot[arm].append({k: float(np.mean([v[k] for v in vals])) for k in vals[0]})
    ci = {arm: {k: [float(np.nanpercentile([x[k] for x in bs], 2.5)), float(np.nanpercentile([x[k] for x in bs], 97.5))]
                for k in bs[0]} for arm, bs in boot.items()}
    diffs = {}
    if all(a in boot for a in OPUS):
        opus = [np.mean([boot[a][b]["open"] for a in OPUS]) for b in range(B)]
        for ref in ("NB", "realpos", "K", "Km", "TB"):
            if ref in boot:
                d = np.array(opus) - np.array([boot[ref][b]["open"] for b in range(B)])
                diffs[f"opus_mean_minus_{ref}"] = {"point": float(np.mean([point[a]["open"] for a in OPUS]) - point[ref]["open"]),
                                                   "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    descr = {}
    for arm, lst in by.items():
        descr[arm] = {c: float(np.mean([np.mean(np.concatenate([s for i, (l, s) in rec.items() if l == c]) > 0) for seed, n, rec in lst]))
                      for c in ("pigeon", "raven") if c in groups}
    out = {"point": point, "ci95": ci, "diffs": diffs, "pigeon_raven_dwell_frac_above0": descr,
           "n_recordings": {c: len(groups[c]) for c in classes}, "seeds_per_arm": {a: len(l) for a, l in by.items()},
           "B": B, "sanity_max_abs_diff_vs_result_json": worst,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    name = f"W18_summary{'_' + args.tag if args.tag else ''}.json"
    (RUNS / name).write_text(json.dumps(out, indent=1))
    for arm in point:
        p, c = point[arm], ci[arm]
        print(f"{arm:8s} open {p['open']:.3f} [{c['open'][0]:.3f},{c['open'][1]:.3f}]  heron {p['heron']:.3f} [{c['heron'][0]:.3f},{c['heron'][1]:.3f}]"
              f"  CR {p['CR']:.3f}  TPR@1/5/10% {p['tpr@0.01']:.2f}/{p['tpr@0.05']:.2f}/{p['tpr@0.10']:.2f}")
    print(json.dumps(diffs, indent=1))


if __name__ == "__main__":
    main()
