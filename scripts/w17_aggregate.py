"""W17 aggregate: pool out-of-fold dwell scores per mode, track20 AUROC per family (same definitions as W14)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from e3w8_sim_rotor_detector import DRONES, TRACK
from w14_train import FAMS, OPEN

RUNS = Path(__file__).resolve().parent.parent / "runs"


def main():
    out = {}
    for mode in ("realpos", "sup"):
        trk = {}
        dirs = sorted(RUNS.glob(f"W17_{mode}_f*_*/oof_scores.json"))
        if len(dirs) < 5:
            continue
        for p in dirs:
            for key, s in json.loads(p.read_text()).items():
                lab = key.split("|", 1)[1]
                s = np.asarray(s)
                nt = len(s) // TRACK
                if nt:
                    trk.setdefault(lab, []).append(s[:nt * TRACK].reshape(nt, TRACK).mean(1))
        pos = np.concatenate([np.concatenate(trk[c]) for c in DRONES])
        r = {}
        for fam, cls in FAMS.items():
            neg = np.concatenate([np.concatenate(trk[c]) for c in cls if c in trk])
            r[fam] = {"auroc": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])),
                      "fa_thr0": float((neg > 0).mean())}
        r["open_auroc_mean"] = float(np.mean([r[f]["auroc"] for f in OPEN]))
        r["detect_thr0"] = {c: float((np.concatenate(trk[c]) > 0).mean()) for c in DRONES}
        out[mode] = r
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (RUNS / "W17_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
