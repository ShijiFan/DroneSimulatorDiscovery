"""E3w W8 (exploratory): rotor-presence detector trained ONLY on simulation, tested zero-shot on real SAAB.
Positives: sim_v3 multirotor returns. Negatives:
  N1  body + noise only (no micro-motion physics for non-targets at all)
  N2  N1 + 'oscillator' targets: 2-8 point scatterers moving back and forth along the line of sight,
      displacement A sin(2 pi f_w t + phase), A 0.02-0.4 m, f_w 1-12 Hz (flapping wings / limbs, generic, not fitted)
Real test (no training on real data): AUROC drones (D1-D6) vs each non-drone family, per segment and per 20-segment track.
Usage: python e3w8_sim_rotor_detector.py [--steps 4000] [--pos v1|v2|v3] [--only_n2]
(the first run, runs/E3w8_20261013_045910, was --pos v3 with both variants, before these options existed)
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
import sim_v2 as S2
import sim_v3 as S3
from train_e0b import Net, feats

DATA = Path(r"E:\rag\datasets\data3\data_SAAB_SIRS_77GHz_FMCW.npy")
DRONES = ["D1", "D2", "D3", "D4", "D5", "D6"]
FAMS = {"gull": ["seagull", "black-headed gull"], "human": ["human_walk", "human_run"], "CR": ["CR"]}
TRACK = 20


def negatives(B, g, dev, with_osc):
    t = torch.arange(S.N, device=dev, dtype=torch.float64) / S.PRF
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=g, device=dev, dtype=torch.float64)
    v_r = u(-5, 5)
    carrier = torch.exp(1j * 2 * math.pi * (2 * v_r / S.LAM)[:, None] * t)
    tc = 7.5e-3 + u(-1e-3, 1e-3)
    w = torch.exp(-4 * math.log(2) * (t[None, :] - tc[:, None]) ** 2 / S.FWHM ** 2)
    body = torch.exp(1j * 2 * math.pi * u(0, 1))[:, None] * carrier * w
    sig = body
    if with_osc:
        K = 8
        A = 0.02 + 0.38 * torch.rand(B, K, generator=g, device=dev, dtype=torch.float64)
        fw = 1 + 11 * torch.rand(B, 1, generator=g, device=dev, dtype=torch.float64)
        ph = 2 * math.pi * torch.rand(B, K, generator=g, device=dev, dtype=torch.float64)
        nk = torch.randint(2, K + 1, (B, 1), generator=g, device=dev)
        mask = (torch.arange(K, device=dev)[None, :] < nk).double()
        amp = torch.complex(torch.randn(B, K, generator=g, device=dev, dtype=torch.float64),
                            torch.randn(B, K, generator=g, device=dev, dtype=torch.float64))
        disp = A[..., None] * torch.sin(2 * math.pi * fw[..., None] * t + ph[..., None])
        osc = (amp[..., None] * mask[..., None] * torch.exp(1j * 4 * math.pi / S.LAM * disp)).sum(1)
        p_b = (body.abs() ** 2).mean(1)
        osc = osc * carrier * w
        rel = 10 ** (-u(0, 25) / 10)
        osc = osc * torch.sqrt(p_b * rel / (osc.abs() ** 2).mean(1).clamp_min(1e-30))[:, None]
        use = (torch.rand(B, generator=g, device=dev) < 0.7).double()[:, None]
        sig = body + use * osc
    p = (sig.abs() ** 2).mean(1)
    snr = u(0, 40)
    noise = torch.sqrt(p / 10 ** (snr / 10) / 2)[:, None] * torch.complex(
        torch.randn(B, S.N, generator=g, device=dev, dtype=torch.float64),
        torch.randn(B, S.N, generator=g, device=dev, dtype=torch.float64))
    x = sig + noise
    return (x / torch.sqrt((x.abs() ** 2).mean(1, keepdim=True))).to(torch.complex64)


def load_real():
    d = np.load(DATA, allow_pickle=True)
    out = []
    for i, r in enumerate(d):
        lab = str(np.squeeze(r[0]))
        X = np.asarray(r[1])[512:768].T
        edge = np.asarray(r[5]).ravel().astype(bool)
        if edge.size == X.shape[0]:
            X = X[~edge]
        if len(X) == 0:
            continue
        X = X / np.sqrt((np.abs(X) ** 2).mean(1, keepdims=True))
        out.append((lab, i, X.astype(np.complex64)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--pos", choices=["v1", "v2", "v3"], default="v3")
    ap.add_argument("--only_n2", action="store_true")
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    dev = "cuda"
    here = Path(__file__).resolve().parent
    out = here.parent / "runs" / f"E3w8_pos{args.pos}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    real = load_real()
    res = {}
    variants = (("N1_body_only", False), ("N2_body_plus_oscillators", True))
    if args.only_n2:
        variants = variants[1:]
    for variant, with_osc in variants:
        torch.manual_seed(args.seed)
        g = torch.Generator(device=dev).manual_seed(args.seed)
        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps, pct_start=0.05)
        t0 = time.time()
        for step in range(args.steps):
            snr = -5 + 30 * torch.rand(args.bs, generator=g, device=dev)
            if args.pos == "v3":
                xp = S3.render_v3(S3.sample_drones_v3(args.bs, g, dev), snr, g, dev)
            elif args.pos == "v2":
                xp = S2.render_v2(S2.sample_drones_v2(args.bs, g, dev), snr, g, dev)
            else:
                xp = S.render(S.sample_drones(args.bs, g, dev), snr, g, dev)
            xn = negatives(args.bs, g, dev, with_osc)
            x = feats(torch.cat([xp, xn]))
            y = torch.cat([torch.ones(args.bs, device=dev), torch.zeros(args.bs, device=dev)])
            loss = F.binary_cross_entropy_with_logits(net(x)[:, 0], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 500 == 0:
                print(variant, step, float(loss), round(time.time() - t0), flush=True)
        torch.save(net.state_dict(), out / f"net_{variant}.pt")
        net.eval()
        seg_scores, trk_scores = {}, {}
        with torch.no_grad():
            for lab, i, X in real:
                s = np.concatenate([net(feats(torch.from_numpy(X[j:j + 2048]).to(dev)))[:, 0].float().cpu().numpy()
                                    for j in range(0, len(X), 2048)])
                seg_scores.setdefault(lab, []).append(s)
                nt = len(s) // TRACK
                if nt:
                    trk_scores.setdefault(lab, []).append(s[:nt * TRACK].reshape(nt, TRACK).mean(1))
        r = {}
        for unit, sc in (("segment", seg_scores), ("track20", trk_scores)):
            pos = np.concatenate([np.concatenate(sc[c]) for c in DRONES if c in sc])
            r[unit] = {"drone_median_logit": float(np.median(pos))}
            for fam, cls in FAMS.items():
                neg = np.concatenate([np.concatenate(sc[c]) for c in cls if c in sc])
                yv = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                r[unit][fam] = {"auroc": float(roc_auc_score(yv, np.r_[pos, neg])), "median_logit": float(np.median(neg)),
                                "n_neg": int(len(neg))}
            r[unit]["per_drone_median_logit"] = {c: float(np.median(np.concatenate(sc[c]))) for c in DRONES if c in sc}
        res[variant] = r
        print(variant, json.dumps(r), flush=True)
    res["args"] = vars(args)
    res["script_sha256"] = {p: hashlib.sha256((here / p).read_bytes()).hexdigest()
                            for p in ["e3w8_sim_rotor_detector.py", "sim_v2.py", "sim_v3.py", "sim_saab.py", "train_e0b.py"]}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(out)


if __name__ == "__main__":
    main()
