"""E0b: amortized inference of (L_eff, f0, N_b) from aliased SAAB-like IQ, vs classical and prior baselines.
See ../PROTOCOL.md E0b. Run with a CUDA torch env."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sim_saab as S
import sim_v2 as S2
import sim_v3 as S3
from rotor import estimate_max_doppler, estimate_period_autocorr

TRAIN_SEED, TEST_SEED = 11, 2026
SNR_TRAIN = (-5.0, 25.0)
SNR_TEST = [0.0, 10.0, 20.0]
K_LOOKS = [1, 4, 16]


def feats(x):
    return torch.stack([x.real, x.imag, x.abs()], 1)


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        self.c2 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return F.gelu(x + self.n2(self.c2(h)))


class Net(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.stem = nn.Conv1d(3, c, 7, padding=3)
        self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 16, 32, 1, 4, 16)])
        self.head = nn.Sequential(nn.Linear(2 * c, 256), nn.GELU(), nn.Linear(256, 4))

    def forward(self, x):
        h = self.blocks(F.gelu(self.stem(x)))
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(h)  # [logL_eff, logf0, nb logits(2)]


SIM = "v1"


def draw(B, g, dev):
    if SIM == "v3":
        return S3.sample_drones_v3(B, g, dev)
    return S2.sample_drones_v2(B, g, dev) if SIM == "v2" else S.sample_drones(B, g, dev)


def rend(d, snr, g, dev):
    if SIM == "v3":
        return S3.render_v3(d, snr, g, dev)
    return S2.render_v2(d, snr, g, dev) if SIM == "v2" else S.render(d, snr, g, dev)


def batch(B, g, dev):
    d = draw(B, g, dev)
    snr = SNR_TRAIN[0] + (SNR_TRAIN[1] - SNR_TRAIN[0]) * torch.rand(B, generator=g, device=dev)
    x = rend(d, snr, g, dev)
    y = torch.stack([d["L_eff"].log(), d["f0"].log()], 1).float()
    return feats(x), y, (d["n_b"] == 3).long()


def classical_one(x, n_b):
    period = estimate_period_autocorr(x, S.PRF, n_b, f_rot_bounds=(20.0, 850.0))
    f0 = 1 / (n_b * period) if np.isfinite(period) else np.nan
    fmax = estimate_max_doppler(x, S.PRF)
    v_tip = fmax * S.LAM / 2
    return (v_tip / (2 * np.pi * f0) if np.isfinite(f0) else np.nan), f0


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--sim", choices=["v1", "v2", "v3"], default="v1")
    args = ap.parse_args()
    global SIM
    SIM = args.sim
    dev = "cuda"
    here = Path(__file__).resolve().parent
    out = here.parent / "runs" / f"E0b{'' if SIM == 'v1' else '_' + SIM}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    torch.manual_seed(TRAIN_SEED)
    g = torch.Generator(device=dev).manual_seed(TRAIN_SEED)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps, pct_start=0.05)
    log = []
    t0 = time.time()
    net.train()
    for step in range(args.steps):
        xb, yb, nb = batch(args.bs, g, dev)
        p = net(xb)
        loss = F.smooth_l1_loss(p[:, :2], yb, beta=0.05) + 0.3 * F.cross_entropy(p[:, 2:], nb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            log.append({"step": step, "loss": float(loss), "elapsed_s": round(time.time() - t0, 1)})
            print(log[-1], flush=True)
    torch.save(net.state_dict(), out / "net.pt")

    # prior-median baseline from the training prior (independent draw)
    gp = torch.Generator(device=dev).manual_seed(999)
    dp = S.sample_drones(200_000, gp, dev)
    prior_L, prior_f = float(dp["L_eff"].median()), float(dp["f0"].median())

    net.eval()
    gt = torch.Generator(device=dev).manual_seed(TEST_SEED)
    rows, summary = [], []
    for snr in SNR_TEST:
        d = draw(args.n_test, gt, dev)
        preds, cls = [], []
        for _ in range(max(K_LOOKS)):
            x = rend(d, torch.full((args.n_test,), snr, device=dev), gt, dev)
            with torch.no_grad():
                preds.append(net(feats(x)).float().cpu())
            xn = x.cpu().numpy()
            nbs = d["n_b"].cpu().numpy()
            cls.append([classical_one(xn[i].astype(np.complex128), int(nbs[i])) for i in range(args.n_test)])
        P = torch.stack(preds, 1).numpy()  # [n,K,4]
        Cc = np.array(cls).transpose(1, 0, 2)  # [n,K,2]
        Le, f0 = d["L_eff"].cpu().numpy(), d["f0"].cpu().numpy()
        nb_true = (d["n_b"].cpu().numpy() == 3)
        for K in K_LOOKS:
            L_hat = np.exp(np.median(P[:, :K, 0], 1))
            f_hat = np.exp(np.median(P[:, :K, 1], 1))
            nb_hat = np.mean(P[:, :K, 3] > P[:, :K, 2], 1) > 0.5
            Lc = np.nanmedian(Cc[:, :K, 0], 1)
            fc = np.nanmedian(Cc[:, :K, 1], 1)
            rel = lambda a, b: np.abs(a / b - 1)
            rec = {"snr_db": snr, "K": K}
            for name, Lh, fh in [("net", L_hat, f_hat), ("classical_trueNb", Lc, fc),
                                 ("prior_median", np.full_like(Le, prior_L), np.full_like(f0, prior_f))]:
                eL, ef = rel(Lh, Le), rel(fh, f0)
                rec[f"{name}_L_med"] = float(np.nanmedian(eL)) if np.isfinite(eL).any() else float("nan")
                rec[f"{name}_L_gross20"] = float(np.mean(~(eL <= 0.20)))
                rec[f"{name}_f_med"] = float(np.nanmedian(ef)) if np.isfinite(ef).any() else float("nan")
                rec[f"{name}_f_gross10"] = float(np.mean(~(ef <= 0.10)))
            rec["net_nb_acc"] = float(np.mean(nb_hat == nb_true))
            summary.append(rec)
            print(rec, flush=True)
        for i in range(args.n_test):
            rows.append({"snr_db": snr, "i": i, "L_eff": Le[i], "f0": f0[i], "n_b": int(2 + nb_true[i]),
                         "n_rot": int(d["n_rot"][i]), "L_hat_K16": float(np.exp(np.median(P[i, :, 0]))),
                         "f_hat_K16": float(np.exp(np.median(P[i, :, 1])))})

    for name, data in [("summary.csv", summary), ("test_drones.csv", rows), ("train_log.csv", log)]:
        with (out / name).open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    gate = [s for s in summary if s["snr_db"] == 10.0 and s["K"] == 16][0]
    manifest = {
        "protocol": "openset_physics/PROTOCOL.md E0b", "train_seed": TRAIN_SEED, "test_seed": TEST_SEED,
        "steps": args.steps, "batch": args.bs, "n_test_per_snr": args.n_test, "prior": S.PRIOR, "prior2": S2.PRIOR2, "prior3": S3.PRIOR3,
        "prior_median": {"L_eff": prior_L, "f0": prior_f},
        "runtime_s": round(time.time() - t0, 1), "python": sys.version, "torch": torch.__version__,
        "script_sha256": {p: sha256(here / p) for p in ["train_e0b.py", "sim_saab.py", "sim_v2.py", "sim_v3.py", "rotor.py"]}, "sim": SIM,
        "outputs_sha256": {p: sha256(out / p) for p in ["summary.csv", "test_drones.csv", "net.pt"]},
        "gate_snr10_K16": {"L_med": gate["net_L_med"], "f_med": gate["net_f_med"]},
        "gate_feasible": bool(gate["net_L_med"] <= 0.10 and gate["net_f_med"] <= 0.05),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(out)
    print(json.dumps(manifest["gate_snr10_K16"]), manifest["gate_feasible"])


if __name__ == "__main__":
    main()
