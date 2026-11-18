"""W18 step 1 (PROTOCOL W18): re-score every saved W14-chain detector (net.pt) on all SAAB recordings and save
per-recording dwell scores, so recording-level statistics can be computed without retraining.
Also used by W19/W20 (pass run-directory globs).  Usage: python w18_rescore.py [glob ...]  (default: all W14_* runs)
Output: runs/W18_scores/<run name>.npz with one array per recording key 'r{index}' plus 'labels' / 'index'."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from e3w8_sim_rotor_detector import load_real
from train_e0b import Net, feats

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT = RUNS / "W18_scores"


def main():
    OUT.mkdir(exist_ok=True)
    globs = sys.argv[1:] or ["W14_*"]
    runs = sorted({p.parent for g in globs for p in RUNS.glob(g + "/net.pt")})
    real = load_real()
    dev = "cuda"
    done = 0
    for run in runs:
        dst = OUT / f"{run.name}.npz"
        if dst.exists():
            continue
        net = Net().to(dev)
        net.load_state_dict(torch.load(run / "net.pt", map_location=dev))
        net.eval()
        arrs = {}
        with torch.no_grad():
            for lab, i, X in real:
                arrs[f"r{i:03d}"] = np.concatenate([net(feats(torch.from_numpy(X[j:j + 2048]).to(dev)))[:, 0].float().cpu().numpy()
                                                    for j in range(0, len(X), 2048)])
        np.savez_compressed(dst, labels=np.array([lab for lab, i, X in real]), index=np.array([i for lab, i, X in real]), **arrs)
        done += 1
        print(run.name, flush=True)
    (OUT / "meta.json").write_text(json.dumps({"n_runs": len(runs), "newly_scored": done,
                                               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1))
    print("scored", done, "of", len(runs))


if __name__ == "__main__":
    main()
