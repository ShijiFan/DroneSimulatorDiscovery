"""W14: train the W8 detector with positives from a fixed rotor bank (one arm), N2 negatives; evaluate zero-shot on SAAB.
Criteria: ../PROTOCOL.md section W14.  Usage: python w14_train.py --arm O --bank <bank_O.npz> --seed 31
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import sim_saab as S
from e3w8_sim_rotor_detector import DRONES, TRACK, load_real
from e3w8b_ablate import negatives
from train_e0b import Net, feats

FAMS = {"gull": ["seagull", "black-headed gull", "seagull and black-headed gull"], "human": ["human_walk", "human_run"],
        "heron": ["heron"], "CR": ["CR"]}
OPEN = ["gull", "human", "heron"]
N_HOLD = 4000  # last N_HOLD bank entries are held out for the closed-set (sim) check


def positives(bank, idx, g, dev):
    B = len(idx)
    t = torch.arange(S.N, device=dev, dtype=torch.float64) / S.PRF
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=g, device=dev, dtype=torch.float64)
    rot = bank[idx].to(torch.complex128)
    v_r = u(-5, 5)
    carrier = torch.exp(1j * 2 * math.pi * (2 * v_r / S.LAM)[:, None] * t)
    tc = 7.5e-3 + u(-1e-3, 1e-3)
    w = torch.exp(-4 * math.log(2) * (t[None, :] - tc[:, None]) ** 2 / S.FWHM ** 2)
    rot = rot * carrier * w
    p = (rot.abs() ** 2).mean(1)
    body = torch.sqrt(p * 10 ** (u(0, 20) / 10))[:, None] * torch.exp(1j * 2 * math.pi * u(0, 1))[:, None] * carrier * w
    sig = rot + body
    ps = (sig.abs() ** 2).mean(1)
    snr = u(0, 40)
    cr = torch.complex(torch.randn(B, S.N, generator=g, device=dev, dtype=torch.float64),
                       torch.randn(B, S.N, generator=g, device=dev, dtype=torch.float64))
    x = sig + torch.sqrt(ps / 10 ** (snr / 10) / 2)[:, None] * cr
    return (x / torch.sqrt((x.abs() ** 2).mean(1, keepdim=True))).to(torch.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    dev = "cuda"
    here = Path(__file__).resolve().parent
    out = here.parent / "runs" / f"W14_{args.arm}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    bank = torch.from_numpy(np.load(args.bank)["x"]).to(dev)
    n_train = len(bank) - N_HOLD
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps, pct_start=0.05)
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, n_train, (args.bs,), generator=g, device=dev)
        xp = positives(bank, idx, g, dev)
        xn = negatives(args.bs, g, dev, 1.0, "N2")
        y = torch.cat([torch.ones(args.bs, device=dev), torch.zeros(args.bs, device=dev)])
        loss = F.binary_cross_entropy_with_logits(net(feats(torch.cat([xp, xn])))[:, 0], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(step, float(loss), round(time.time() - t0), flush=True)
    torch.save(net.state_dict(), out / "net.pt")
    net.eval()
    res = {}
    with torch.no_grad():
        gh = torch.Generator(device=dev).manual_seed(999)
        hp = positives(bank, torch.arange(n_train, len(bank), device=dev), gh, dev)
        hn = negatives(N_HOLD, gh, dev, 1.0, "N2")
        sc = torch.cat([net(feats(hp[j:j + 2000]))[:, 0] for j in range(0, N_HOLD, 2000)] +
                       [net(feats(hn[j:j + 2000]))[:, 0] for j in range(0, N_HOLD, 2000)]).float().cpu().numpy()
        res["closed_sim_auroc"] = float(roc_auc_score(np.r_[np.ones(N_HOLD), np.zeros(N_HOLD)], sc))
        seg, trk = {}, {}
        for lab, i, X in load_real():
            s = np.concatenate([net(feats(torch.from_numpy(X[j:j + 2048]).to(dev)))[:, 0].float().cpu().numpy()
                                for j in range(0, len(X), 2048)])
            seg.setdefault(lab, []).append(s)
            nt = len(s) // TRACK
            if nt:
                trk.setdefault(lab, []).append(s[:nt * TRACK].reshape(nt, TRACK).mean(1))
    for unit, sc in (("segment", seg), ("track20", trk)):
        pos = np.concatenate([np.concatenate(sc[c]) for c in DRONES])
        r = {"drone_median_logit": float(np.median(pos))}
        for fam, cls in FAMS.items():
            neg = np.concatenate([np.concatenate(sc[c]) for c in cls if c in sc])
            r[fam] = {"auroc": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])),
                      "fa_thr0": float((neg > 0).mean()), "n_neg": int(len(neg))}
        r["open_auroc_mean"] = float(np.mean([r[f]["auroc"] for f in OPEN]))
        r["detect_thr0"] = {c: float((np.concatenate(sc[c]) > 0).mean()) for c in DRONES}
        res[unit] = r
    res["args"] = vars(args)
    res["train_s"] = round(time.time() - t0)
    res["bank_sha256"] = hashlib.sha256(Path(args.bank).read_bytes()).hexdigest()
    res["script_sha256"] = {p: hashlib.sha256((here / p).read_bytes()).hexdigest()
                            for p in ["w14_train.py", "e3w8b_ablate.py", "e3w8_sim_rotor_detector.py", "sim_saab.py", "train_e0b.py"]}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"closed_sim_auroc": res["closed_sim_auroc"], **res["track20"]}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
