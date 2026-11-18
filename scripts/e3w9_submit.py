"""W9 (PROTOCOL.md E4/W9): submission gate for a Claude sub-agent acting as the LLM in the W7 forward-model loop.
Evaluates one candidate with the unchanged W7 evaluator (seed 2026), prints the same feedback text qwen got in W7,
enforces the 20-submission budget, and logs every submission inside the sandbox.
Usage: python e3w9_submit.py <sandbox dir> <candidate.py>
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL_NPZ = HERE.parent / "runs" / "W9_claude_20261013_124339" / "real_shape.npz"
REAL_ASYM = -4.419517287189267  # runs/E3w7_20261013_004649/blind/log.json
BUDGET = 20


def main():
    box, cand = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    logf = box / "submissions.json"
    log = json.loads(logf.read_text()) if logf.exists() else []
    if len(log) >= BUDGET:
        print(f"Budget exhausted: {BUDGET} submissions already evaluated. Stop and write your final answer.")
        return
    k = len(log) + 1
    dst = box / "subs" / f"sub_{k:02d}.py"
    dst.parent.mkdir(exist_ok=True)
    shutil.copyfile(cand, dst)
    out = dst.with_suffix(".json")
    try:
        p = subprocess.run([sys.executable, "-I", str(HERE / "e3w7_eval.py"), str(dst), str(REAL_NPZ), str(out), "2026"],
                           capture_output=True, text=True, timeout=180)
        err = None if p.returncode == 0 else (p.stderr.strip().splitlines()[-1][:400] if p.stderr.strip() else "unknown error")
    except subprocess.TimeoutExpired:
        err = "evaluation timed out (180 s for 300 calls); make the function faster"
    rec = {"sub": k, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "error": err}
    if err is None:
        res = json.loads(out.read_text())
        rec.update(fitness=res["fitness_rms_db"], sim_asym=res["sim_asym_db_hump"])
        pairs = ", ".join(f"{c:+.1f}kHz:{v:+.1f}" for c, v in zip(res["band_centres_khz"], res["sim_minus_real_db"])
                          if v is not None)
        msg = (f"Submission {k}/{BUDGET}. Fitness {res['fitness_rms_db']:.2f} dB. Per 1-kHz band, simulated minus measured "
               f"residual level (dB, floor-referenced; positive = your model has too much energy there): {pairs}. "
               f"Measured hump asymmetry (P(+0.25..3 kHz)/P(-3..-0.25 kHz)) = {REAL_ASYM:+.1f} dB; yours = "
               f"{res['sim_asym_db_hump']:+.1f} dB.")
    else:
        msg = f"Submission {k}/{BUDGET} failed: {err}"
    log.append(rec)
    logf.write_text(json.dumps(log, indent=1))
    print(msg)


if __name__ == "__main__":
    main()
