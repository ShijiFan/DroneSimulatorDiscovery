"""W16 aggregate: human/expert baselines (TB, K, NB) vs the Opus-source group (O, O2, O3, O4, D); PROTOCOL W16."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from w14b_aggregate import model_stats

RUNS = Path(__file__).resolve().parent.parent / "runs"
B = RUNS / "W16_baselines"
LLM = ["O", "O2", "O3", "O4", "D"]
HUM = ["TB", "K", "NB"]


def main():
    st = {a: model_stats(a) for a in LLM + HUM}
    M = json.loads((B / "metric_M.json").read_text())
    fit = {a: float(np.mean([json.loads((B / f"fit_{a}_{s}.json").read_text())["fitness_rms_db"] for s in (101, 202, 303)]))
           for a in HUM}
    llm_mean = float(np.mean([st[a]["open"] for a in LLM]))
    best_h = max(HUM, key=lambda a: st[a]["open"])
    out = {"models": {a: {**st[a], **({"fit_db": fit[a], "M": M[a]["M"]} if a in HUM else {})} for a in LLM + HUM},
           "llm_mean_open": llm_mean, "best_human": best_h, "best_human_open": st[best_h]["open"],
           "C1_pass": bool(llm_mean - st[best_h]["open"] >= 0.01), "C2_K_minus_llm": st["K"]["open"] - llm_mean,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (RUNS / "W16_summary.json").write_text(json.dumps(out, indent=1))
    for a in LLM + HUM:
        v = st[a]
        print("%-3s open %.4f gull %.3f human %.3f heron %.3f closed %.3f D2-6 %.2f D1 %.2f heronFA %.2f %s" %
              (a, v["open"], v["gull"], v["human"], v["heron"], v["closed"], v["D2_D6"], v["D1"], v["heron_fa"],
               ("fit %.2f M %.2f" % (fit[a], M[a]["M"])) if a in HUM else ""))
    print(json.dumps({k: out[k] for k in ("llm_mean_open", "best_human", "best_human_open", "C1_pass", "C2_K_minus_llm")}, indent=1))


if __name__ == "__main__":
    main()
