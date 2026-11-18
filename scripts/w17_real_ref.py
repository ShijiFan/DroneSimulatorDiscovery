"""W17: real-data reference detectors on SAAB (PROTOCOL W17), 5-fold CV grouped by recording.
mode realpos: positives = real drone dwells of training-fold recordings, negatives = W8 N2 synthetic oscillators.
mode sup:     positives = real drones, negatives = real gull/human/CR/pigeon/raven dwells (training folds); heron never trained.
Out-of-fold dwell scores are saved; w17_aggregate.py pools them.  Usage: python w17_real_ref.py --mode sup --fold 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from e3w8_sim_rotor_detector import DRONES, load_real
from e3w8b_ablate import negatives
from train_e0b import Net, feats

K = 5
NEG_REAL = ["seagull", "black-headed gull", "seagull and black-headed gull", "human_walk", "human_run", "CR", "pigeon", "raven"]


def folds(real):
    f, byl = {}, {}
    for lab, i, X in real:
        byl.setdefault(lab, []).append(i)
    for lab, ids in byl.items():
        for r, i in enumerate(sorted(ids)):
            f[i] = r % K
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["realpos", "sup"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    dev = "cuda"
    here = Path(__file__).resolve().parent
    out = here.parent / "runs" / f"W17_{args.mode}_f{args.fold}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    real = load_real()
    fo = folds(real)
    tr = [(lab, i, X) for lab, i, X in real if fo[i] != args.fold]
    te = [(lab, i, X) for lab, i, X in real if fo[i] == args.fold]
    P = torch.from_numpy(np.concatenate([X for lab, i, X in tr if lab in DRONES])).to(dev)
    N_ = torch.from_numpy(np.concatenate([X for lab, i, X in tr if lab in NEG_REAL])).to(dev) if args.mode == "sup" else None
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps, pct_start=0.05)
    t0 = time.time()
    for step in range(args.steps):
        xp = P[torch.randint(0, len(P), (args.bs,), generator=g, device=dev)]
        xn = N_[torch.randint(0, len(N_), (args.bs,), generator=g, device=dev)] if N_ is not None else \
            negatives(args.bs, g, dev, 1.0, "N2")
        y = torch.cat([torch.ones(args.bs, device=dev), torch.zeros(args.bs, device=dev)])
        loss = F.binary_cross_entropy_with_logits(net(feats(torch.cat([xp, xn])))[:, 0], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(step, loss.item(), round(time.time() - t0), flush=True)
    net.eval()
    scores = {}
    with torch.no_grad():
        for lab, i, X in te:
            s = np.concatenate([net(feats(torch.from_numpy(X[j:j + 2048]).to(dev)))[:, 0].float().cpu().numpy()
                                for j in range(0, len(X), 2048)])
            scores[f"{i}|{lab}"] = s.tolist()
    (out / "oof_scores.json").write_text(json.dumps(scores))
    (out / "meta.json").write_text(json.dumps({"args": vars(args), "n_train_pos": int(len(P)),
                                               "n_train_neg_real": int(len(N_)) if N_ is not None else 0,
                                               "train_s": round(time.time() - t0),
                                               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
