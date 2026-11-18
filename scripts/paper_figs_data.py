"""Paper: dataset table (recordings / dwells / duration per class) and Fig. real per-family residual spectra,
plus Fig. heron score timeline (from W13 per-dwell scores).  Usage: python paper_figs_data.py <out_dir>"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import e3w7_eval as EV
from e3w8_sim_rotor_detector import DATA

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
R = Path(__file__).resolve().parent.parent / "runs"
FAM = {"Drones (D1-D6)": ["D1", "D2", "D3", "D4", "D5", "D6"], "Gulls": ["seagull", "black-headed gull", "seagull and black-headed gull"],
       "Heron": ["heron"], "Humans": ["human_walk", "human_run"], "Corner reflector": ["CR"]}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA, allow_pickle=True)
    rows, per = {}, {}
    for r in d:
        lab = str(np.squeeze(r[0]))
        X = np.asarray(r[1])[512:768].T
        t = np.asarray(r[3]).ravel()
        e = np.asarray(r[5]).ravel().astype(bool)
        if e.size == len(X):
            X, t = X[~e], t[~e]
        s = rows.setdefault(lab, {"recordings": 0, "dwells": 0, "duration_s": 0.0})
        s["recordings"] += 1
        s["dwells"] += len(X)
        s["duration_s"] += float(t[-1] - t[0]) if len(t) > 1 else 0.0
        per.setdefault(lab, []).append(X)
    with (out / "table_dataset.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["class", "recordings", "dwells", "duration_s"])
        for k, v in sorted(rows.items()):
            w.writerow([k, v["recordings"], v["dwells"], round(v["duration_s"], 1)])
    rng = np.random.default_rng(0)
    f = np.fft.fftshift(np.fft.fftfreq(EV.NF, 1 / EV.PRF))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    sty = {"Drones (D1-D6)": ("k", "-"), "Gulls": ("#1f77b4", "--"), "Heron": ("#d62728", "-."),
           "Humans": ("#2ca02c", ":"), "Corner reflector": ("0.5", "-")}
    spec_meta = {}
    for name, cls in FAM.items():
        X = np.concatenate([x for c in cls for x in per.get(c, [])])
        idx = rng.choice(len(X), min(3000, len(X)), replace=False)
        P = np.mean([EV.spec(X[i].astype(complex)) for i in idx], 0)
        s = EV.shape(P, f)
        c, ls = sty[name]
        ax.plot(f / 1e3, s, color=c, ls=ls, lw=1.1, label=f"{name} (n={len(idx)})")
        spec_meta[name] = int(len(idx))
    ax.axvspan(-0.25, 0.25, color="0.9", zorder=0)
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-6, 30)
    ax.set_xlabel("Doppler frequency (kHz)")
    ax.set_ylabel("Residual level re floor (dB)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_real_spectra.png", dpi=600)
    fig.savefig(out / "fig_real_spectra.pdf")
    # heron timeline from W13 per-dwell scores (seed-31 W8 detector)
    w13 = sorted(R.glob("W13_temporal_*"))[-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for p in sorted(glob.glob(str(w13 / "scores_rec*_heron.npy"))):
        a = np.load(p)
        ax.plot(a[:, 0] - a[0, 0], a[:, 1], lw=0.8, alpha=0.85, label="rec " + re.search(r"rec(\d+)", p).group(1))
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(0, 2.0, color="0.92", zorder=0)
    ax.set_xlabel("Time since track start (s)")
    ax.set_ylabel("Detector logit")
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_heron_timeline.png", dpi=600)
    fig.savefig(out / "fig_heron_timeline.pdf")
    meta = {"table_dataset": rows, "spectra_n_dwells": spec_meta, "w13_dir": str(w13),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "figs_data_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
