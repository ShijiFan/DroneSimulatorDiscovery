"""Paper: track-level drone detection vs heron false alarms at several logit thresholds for the W14_O_s31 detector
(the detector of the heron-timeline figure).  Writes ../paper_RS/figs/heron_threshold.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from e3w8_sim_rotor_detector import DATA, DRONES
from train_e0b import Net, feats

R = Path(__file__).resolve().parent.parent / "runs"
OUT = Path(__file__).resolve().parent.parent / "paper_RS" / "figs" / "heron_threshold.json"


def main():
    run = sorted(R.glob("W14_O_s31_*"))[-1]
    net = Net().cuda()
    net.load_state_dict(torch.load(run / "net.pt", map_location="cuda"))
    net.eval()
    pos, her = [], []
    for r in np.load(DATA, allow_pickle=True):
        lab = str(np.squeeze(r[0]))
        if lab != "heron" and lab not in DRONES:
            continue
        X = np.asarray(r[1])[512:768].T
        e = np.asarray(r[5]).ravel().astype(bool)
        if e.size == len(X):
            X = X[~e]
        X = (X / np.sqrt((np.abs(X) ** 2).mean(1, keepdims=True))).astype(np.complex64)
        with torch.no_grad():
            s = np.concatenate([net(feats(torch.from_numpy(X[j:j + 2048]).cuda()))[:, 0].float().cpu().numpy()
                                for j in range(0, len(X), 2048)])
        nt = len(s) // 20
        (her if lab == "heron" else pos).append(s[:nt * 20].reshape(nt, 20).mean(1))
    pos, her = np.concatenate(pos), np.concatenate(her)
    out = {"run": str(run), "n_drone_tracks": int(len(pos)), "n_heron_tracks": int(len(her)),
           "thresholds": {str(th): {"det": float((pos > th).mean()), "heron_fa": float((her > th).mean())}
                          for th in (0.0, 0.5, 1.0, 1.5)},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
