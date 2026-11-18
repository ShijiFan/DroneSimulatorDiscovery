"""E4 (PROTOCOL.md): W8 ablations, sim-only rotor detector tested zero-shot on real SAAB.
  --osc_scale s   oscillator negatives with A in (0.02, 0.4 s) m and f_w in (1, 12 s) Hz      (W8-R; s=1 is W8 N2)
  --neg N3        half of the micro-motion negatives become flapping wings: hinged line scatterers, bounded angular
                  swing theta(t) = Theta sin(2 pi f_w t) (Theta 15-70 deg, f_w 1-12 Hz, span 0.1-0.8 m) + body heave
                  (W8-N3)
  --prior wide    every PRIOR2 interval widened by 50% of its width on each side, clipped at physical bounds (W8-W)
  --neg LLM --bank <npz>   (W12) half of all negatives are N2, half are drawn from an LLM red-team echo bank (generator
                  chosen uniformly, then an instance) and passed through the w12_common chain (bulk v_r, beam, SNR 0-40 dB)
Positives are always sim_v2. Output: result.json (AUROC) and threshold0.json (logit-0 rates), like e3w8 + e3w8_threshold.
Usage: python e3w8b_ablate.py --tag NAME [--osc_scale 1] [--neg N2|N3] [--prior v2|wide] [--seed 31] [--steps 4000]
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
from e3w8_sim_rotor_detector import DRONES, TRACK, load_real
from train_e0b import Net, feats
import w12_common as W12

FAMS = {"gull": ["seagull", "black-headed gull", "seagull and black-headed gull"], "human": ["human_walk", "human_run"],
        "heron": ["heron"], "CR": ["CR"]}
LOWER = {"a_spec_db": None, "alpha": 0.0, "gamma": 0.0, "gamma_hub": 0.0, "hub_db": None, "r_hub": 0.001,
         "body_db": None, "vib_amp": 0.0, "vib_f": 1.0, "spur_f": 500.0, "spur_db": None}


def widen_prior():
    for k, lo_b in LOWER.items():
        lo, hi = S2.PRIOR2[k]
        w = hi - lo
        lo2 = lo - 0.5 * w
        S2.PRIOR2[k] = (lo2 if lo_b is None else max(lo2, lo_b), hi + 0.5 * w)


def negatives(B, g, dev, osc_scale, neg, bank=None):
    t = torch.arange(S.N, device=dev, dtype=torch.float64) / S.PRF
    u = lambda lo, hi, *sh: lo + (hi - lo) * torch.rand(*(sh or (B,)), generator=g, device=dev, dtype=torch.float64)
    cr = lambda *sh: torch.complex(torch.randn(*sh, generator=g, device=dev, dtype=torch.float64),
                                   torch.randn(*sh, generator=g, device=dev, dtype=torch.float64))
    k4 = 4 * math.pi / S.LAM
    v_r = u(-5, 5)
    carrier = torch.exp(1j * 2 * math.pi * (2 * v_r / S.LAM)[:, None] * t)
    tc = 7.5e-3 + u(-1e-3, 1e-3)
    w = torch.exp(-4 * math.log(2) * (t[None, :] - tc[:, None]) ** 2 / S.FWHM ** 2)
    body = torch.exp(1j * 2 * math.pi * u(0, 1))[:, None] * carrier * w
    K = 8
    # reciprocating oscillators (W8 N2)
    A = u(0.02, 0.4 * osc_scale, B, K)
    fw = u(1, 12 * osc_scale, B, 1)
    ph = u(0, 2 * math.pi, B, K)
    nk = torch.randint(2, K + 1, (B, 1), generator=g, device=dev)
    mask = (torch.arange(K, device=dev)[None, :] < nk).double()
    disp = A[..., None] * torch.sin(2 * math.pi * fw[..., None] * t + ph[..., None])
    mm = (cr(B, K) * mask)[..., None] * torch.exp(1j * k4 * disp)
    osc = mm.sum(1)
    if neg == "N3":
        # flapping wings: 2 wings x K points on a hinged line, bounded angular swing about the shoulder
        span = u(0.1, 0.8)
        Th = torch.deg2rad(u(15, 70))
        th0 = torch.deg2rad(u(-20, 20))
        fwf = u(1, 12)
        ph0 = u(0, 2 * math.pi)
        heave = u(0, 0.05)
        th = th0[:, None] + Th[:, None] * torch.sin(2 * math.pi * fwf[:, None] * t + ph0[:, None])  # [B,N]
        hv = heave[:, None] * torch.sin(2 * math.pi * fwf[:, None] * t + ph0[:, None] + math.pi / 2)
        r = span[:, None] * (torch.arange(1, K + 1, device=dev, dtype=torch.float64) / K)[None, :]  # [B,K]
        flap = torch.zeros(B, S.N, dtype=torch.complex128, device=dev)
        for side in (1.0, -1.0):
            asp = torch.cos(u(0, math.radians(80)))  # projection of the wing plane on the line of sight
            los = side * asp[:, None, None] * r[..., None] * torch.sin(th)[:, None, :] + hv[:, None, :]
            flap = flap + (cr(B, K)[..., None] * torch.exp(1j * k4 * los)).sum(1)
        pick = (torch.rand(B, generator=g, device=dev) < 0.5).double()[:, None]
        osc = pick * flap + (1 - pick) * osc
    p_b = (body.abs() ** 2).mean(1)
    osc = osc * carrier * w
    rel = 10 ** (-u(0, 25) / 10)
    osc = osc * torch.sqrt(p_b * rel / (osc.abs() ** 2).mean(1).clamp_min(1e-30))[:, None]
    use = (torch.rand(B, generator=g, device=dev) < 0.7).double()[:, None]
    sig = body + use * osc
    p = (sig.abs() ** 2).mean(1)
    snr = u(0, 40)
    x = sig + torch.sqrt(p / 10 ** (snr / 10) / 2)[:, None] * cr(B, S.N)
    x = (x / torch.sqrt((x.abs() ** 2).mean(1, keepdim=True))).to(torch.complex64)
    if neg == "LLM":
        gi = torch.randint(0, bank["n_gen"], (B,), generator=g, device=dev)
        ii = torch.randint(0, bank["per_gen"], (B,), generator=g, device=dev)
        xl = W12.chain(bank["x"][gi * bank["per_gen"] + ii].to(torch.complex128), g)
        pick = torch.rand(B, generator=g, device=dev) < 0.5
        x = torch.where(pick[:, None], xl, x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--osc_scale", type=float, default=1.0)
    ap.add_argument("--neg", choices=["N2", "N3", "LLM"], default="N2")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--prior", choices=["v2", "wide"], default="v2")
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    dev = "cuda"
    here = Path(__file__).resolve().parent
    prefix = "W12" if args.neg == "LLM" else "E4"
    out = here.parent / "runs" / f"{prefix}_{args.tag}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    bank = None
    if args.neg == "LLM":
        z = np.load(args.bank)
        bank = {"x": torch.from_numpy(z["x"]).to(dev), "n_gen": int(z["n_gen"]), "per_gen": int(z["per_gen"])}
    if args.prior == "wide":
        widen_prior()
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps, pct_start=0.05)
    t0 = time.time()
    for step in range(args.steps):
        snr = -5 + 30 * torch.rand(args.bs, generator=g, device=dev)
        xp = S2.render_v2(S2.sample_drones_v2(args.bs, g, dev), snr, g, dev)
        xn = negatives(args.bs, g, dev, args.osc_scale, args.neg, bank)
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
    seg, trk = {}, {}
    with torch.no_grad():
        for lab, i, X in load_real():
            s = np.concatenate([net(feats(torch.from_numpy(X[j:j + 2048]).to(dev)))[:, 0].float().cpu().numpy()
                                for j in range(0, len(X), 2048)])
            seg.setdefault(lab, []).append(s)
            nt = len(s) // TRACK
            if nt:
                trk.setdefault(lab, []).append(s[:nt * TRACK].reshape(nt, TRACK).mean(1))
    res = {}
    for unit, sc in (("segment", seg), ("track20", trk)):
        pos = np.concatenate([np.concatenate(sc[c]) for c in DRONES])
        r = {"drone_median_logit": float(np.median(pos))}
        for fam, cls in FAMS.items():
            neg = np.concatenate([np.concatenate(sc[c]) for c in cls if c in sc])
            yv = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            r[fam] = {"auroc": float(roc_auc_score(yv, np.r_[pos, neg])), "fa_thr0": float((neg > 0).mean()),
                      "n_neg": int(len(neg))}
        r["detect_thr0"] = {c: float((np.concatenate(sc[c]) > 0).mean()) for c in DRONES}
        r["frac_rotor_thr0_per_class"] = {c: float((np.concatenate(v) > 0).mean()) for c, v in sc.items()}
        res[unit] = r
    res["args"] = vars(args)
    res["prior2_used"] = {k: list(v) if isinstance(v, tuple) else v for k, v in S2.PRIOR2.items()}
    res["train_s"] = round(time.time() - t0)
    res["script_sha256"] = {p: hashlib.sha256((here / p).read_bytes()).hexdigest()
                            for p in ["e3w8b_ablate.py", "e3w8_sim_rotor_detector.py", "sim_v2.py", "sim_saab.py",
                                      "train_e0b.py", "w12_common.py"]}
    if args.bank:
        res["bank_sha256"] = hashlib.sha256(Path(args.bank).read_bytes()).hexdigest()
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["track20"], indent=1))
    print(out)


if __name__ == "__main__":
    main()
