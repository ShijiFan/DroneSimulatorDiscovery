"""W14b aggregate: PROTOCOL W14b criteria over 4 Opus vs 4 Sonnet models (+ D, V3 for the fit-vs-downstream Spearman)."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RUNS = Path(__file__).resolve().parent.parent / "runs"
W9 = RUNS / "W9_claude_20261013_124339"
SEEDS = [31, 32, 33]
OPUS = {"O": "opus_blind_r2", "O2": "opus_blind_r1", "O3": "opus_informed_r1", "O4": "opus_informed_r2"}
SONNET = {"S": "sonnet_blind_r1", "S2": "sonnet_blind_r2", "S3": "sonnet_informed_r1", "S4": "sonnet_informed_r2"}


def model_stats(arm):
    rs = []
    for s in SEEDS:
        hits = sorted(RUNS.glob(f"W14_{arm}_s{s}_*/result.json"))
        assert hits, f"missing {arm} s{s}"
        rs.append(json.loads(hits[-1].read_text()))
    m = lambda f: float(np.mean([f(x) for x in rs]))
    return {"open": m(lambda x: x["track20"]["open_auroc_mean"]), "gull": m(lambda x: x["track20"]["gull"]["auroc"]),
            "human": m(lambda x: x["track20"]["human"]["auroc"]), "heron": m(lambda x: x["track20"]["heron"]["auroc"]),
            "closed": m(lambda x: x["closed_sim_auroc"]),
            "D2_D6": m(lambda x: np.mean([x["track20"]["detect_thr0"][c] for c in ["D2", "D3", "D4", "D5", "D6"]])),
            "D1": m(lambda x: x["track20"]["detect_thr0"]["D1"]), "heron_fa": m(lambda x: x["track20"]["heron"]["fa_thr0"]),
            "per_seed_open": [x["track20"]["open_auroc_mean"] for x in rs]}


def main():
    score = json.loads((W9 / "score.json").read_text())
    fit = {a: score[d]["fresh_mean"] for a, d in {**OPUS, **SONNET}.items()}
    fit["D"] = json.loads((W9 / "distill_opus" / "fit" / "result.json").read_text())["fresh_mean"]
    fit["V3"] = 0.98  # W9 REPORT_ZH.md table (hand v3, fresh seeds 0.92/1.02/0.96)
    st = {a: model_stats(a) for a in list(OPUS) + list(SONNET) + ["D", "V3"]}
    o = np.array([st[a]["open"] for a in OPUS])
    s = np.array([st[a]["open"] for a in SONNET])
    wins = int(sum(oi > si for oi, si in itertools.product(o, s)))
    diff = float(o.mean() - s.mean())
    arms = list(st)
    rho, p = spearmanr([fit[a] for a in arms], [st[a]["open"] for a in arms])
    out = {"models": {a: {"fit_db": fit[a], **st[a]} for a in arms},
           "primary": {"opus_mean": float(o.mean()), "sonnet_mean": float(s.mean()), "diff": diff, "pair_wins_of_16": wins,
                       "pass": bool(diff >= 0.02 and wins >= 15)},
           "spearman_fit_vs_open_10models": {"rho": float(rho), "p": float(p)}}
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (RUNS / "W14b_summary.json").write_text(json.dumps(out, indent=1))
    for a in arms:
        v = st[a]
        print("%-3s fit %.2f open %.4f gull %.3f human %.3f heron %.3f closed %.3f D2-6 %.2f D1 %.2f heronFA %.2f seeds %s" %
              (a, fit[a], v["open"], v["gull"], v["human"], v["heron"], v["closed"], v["D2_D6"], v["D1"], v["heron_fa"],
               np.round(v["per_seed_open"], 3)))
    print(json.dumps({k: out[k] for k in ("primary", "spearman_fit_vs_open_10models")}, indent=1))


if __name__ == "__main__":
    main()
