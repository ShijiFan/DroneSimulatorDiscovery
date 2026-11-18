"""Build the Remote Sensing manuscript from saved results: figures, tables, numbers, citations, docx + LaTeX body.
All numbers come from runs/*_summary.json, W9 score.json and per-run result.json files.
Usage: python paper_build.py   (writes into ../paper_RS)"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from e3w8_sim_rotor_detector import DATA
from train_e0b import Net, feats
from w14b_aggregate import model_stats

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
HERE = Path(__file__).resolve().parent
R = HERE.parent / "runs"
P = HERE.parent / "paper_RS"
FIG = P / "figs"
W9 = R / "W9_claude_20261013_124339"
PANDOC = r"D:\netload\anaconda\Library\bin\pandoc.exe"

NAME = {"O2": "O-b1", "O": "O-b2", "O3": "O-i1", "O4": "O-i2", "D": "O-b1-dist",
        "S": "S-b1", "S2": "S-b2", "S3": "S-i1", "S4": "S-i2", "TB": "Textbook", "K": "Karlsson", "NB": "Hand-fitted",
        "E1": "O-b2 + drift", "E2": "S-b1 − drift", "E3": "S-b2 − tone", "E4": "S-i2 − shift"}
LBL = {"S4": (4, -8), "S2": (-24, 1), "E4": (4, 3)}  # label offsets (points) to avoid overlaps in fig_predictors
OPUS = ["O2", "O", "O3", "O4", "D"]
SONNET = ["S", "S2", "S3", "S4"]
HUMAN = ["TB", "K", "NB"]
EDITS = ["E1", "E2", "E3", "E4"]
DESC = {"O2": "hub points and hub-shadowed root points, 1–11 mm",
        "O": "hub/root points at r ≤ 0.1L with angle-dependent amplitude",
        "O3": "motor-bell points, 2–14 mm, angle-dependent amplitude",
        "O4": "motor-bell points, 0.8–9.5 mm, angle-dependent amplitude",
        "D": "6-parameter family distilled from O-b1",
        "S": "full line blades + point drifting at −0.3 to −4 m/s",
        "S2": "full line blades + small hub term + tone at −0.5 to −4.5 kHz",
        "S3": "root-weighted blades + blade-pass hub modulation + random rotor offsets",
        "S4": "full line blades + tip flash + −800 Hz shift of rotor return",
        "TB": "rigid line blades of radius L [@martin1990analysis]",
        "K": "eqs. (6.4)–(6.8) and Table 6.4 of [@karlsson2023radar]",
        "NB": "line blades + diffuse blade points + hub points, 6 fitted parameters"}
AGENT = {"O2": ("Opus", "blind"), "O": ("Opus", "blind"), "O3": ("Opus", "informed"), "O4": ("Opus", "informed"),
         "D": ("Opus", "distilled"), "S": ("Sonnet", "blind"), "S2": ("Sonnet", "blind"), "S3": ("Sonnet", "informed"),
         "S4": ("Sonnet", "informed"), "TB": ("—", "hand"), "K": ("—", "hand"), "NB": ("—", "hand")}


def fits():
    sc = json.loads((W9 / "score.json").read_text())
    m = {"O2": "opus_blind_r1", "O": "opus_blind_r2", "O3": "opus_informed_r1", "O4": "opus_informed_r2",
         "S": "sonnet_blind_r1", "S2": "sonnet_blind_r2", "S3": "sonnet_informed_r1", "S4": "sonnet_informed_r2"}
    f = {a: sc[d]["fresh_mean"] for a, d in m.items()}
    f["D"] = json.loads((W9 / "distill_opus/fit/result.json").read_text())["fresh_mean"]
    for a in HUMAN:
        f[a] = float(np.mean([json.loads((R / f"W16_baselines/fit_{a}_{s}.json").read_text())["fitness_rms_db"] for s in (101, 202, 303)]))
    ed = {"E1": "E1_O_plus_drift", "E2": "E2_S_minus_drift", "E3": "E3_S2_minus_tone", "E4": "E4_S4_minus_shift"}
    for a, n in ed.items():
        f[a] = float(np.mean([json.loads((R / f"W14c_edits/fit_{n}_{s}.json").read_text())["fitness_rms_db"] for s in (101, 202, 303)]))
    return f


def md_table(caption, header, rows, label):
    s = f"Table: {caption} {{#{label}}}\n\n| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def heron_timeline():
    run = sorted(R.glob("W14_O_s31_*"))[-1]
    net = Net().cuda()
    net.load_state_dict(torch.load(run / "net.pt", map_location="cuda"))
    net.eval()
    d = np.load(DATA, allow_pickle=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, r in enumerate(d):
        if str(np.squeeze(r[0])) != "heron":
            continue
        X = np.asarray(r[1])[512:768].T
        t = np.asarray(r[3]).ravel()
        e = np.asarray(r[5]).ravel().astype(bool)
        if e.size == len(X):
            X, t = X[~e], t[~e]
        X = (X / np.sqrt((np.abs(X) ** 2).mean(1, keepdims=True))).astype(np.complex64)
        with torch.no_grad():
            s = net(feats(torch.from_numpy(X).cuda()))[:, 0].float().cpu().numpy()
        ax.plot(t - t[0], s, lw=0.8, alpha=0.85, label=f"recording {i}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Time since track start (s)")
    ax.set_ylabel("Detector logit")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig_heron_timeline.png", dpi=600)
    fig.savefig(FIG / "fig_heron_timeline.pdf")
    plt.close(fig)
    return str(run)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    F = fits()
    st = {a: model_stats(a) for a in OPUS + SONNET + HUMAN + EDITS}
    w17 = json.loads((R / "W17_summary.json").read_text())
    sims = OPUS + SONNET + HUMAN + EDITS
    open_ = {a: st[a]["open"] for a in sims}
    closed = {a: st[a]["closed"] for a in sims}
    rho, pv = spearmanr([F[a] for a in sims], [open_[a] for a in sims])
    fine = [a for a in sims if F[a] < 2.5]
    rho_f, pv_f = spearmanr([F[a] for a in fine], [open_[a] for a in fine])
    # ---- Fig predictors
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    groups = [("LLM (Opus)", OPUS, "o", "k"), ("LLM (Sonnet)", SONNET, "s", "#d62728"),
              ("Hand-designed", HUMAN, "^", "#1f77b4"), ("Edited programs", EDITS, "D", "0.55")]
    for ax, xs, xl in ((axs[0], closed, "In-simulation AUROC"), (axs[1], F, "Spectral fitness F (dB, lower is better)")):
        for lab, g, mk, c in groups:
            ax.scatter([xs[a] for a in g], [open_[a] for a in g], marker=mk, s=26, facecolor="none" if mk != "o" else c,
                       edgecolor=c, label=lab, zorder=3)
            for a in g:
                if a in OPUS:  # tight cluster near AUROC 0.98; identified by the legend
                    continue
                ax.annotate(NAME[a], (xs[a], open_[a]), fontsize=6, xytext=LBL.get(a, (3, 2)), textcoords="offset points")
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Real open-set AUROC")
    axs[0].set_title("(a)", loc="left", fontsize=10)
    axs[1].set_title("(b)", loc="left", fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.tight_layout()
    fig.savefig(FIG / "fig_predictors.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig_predictors.pdf")
    plt.close(fig)
    # ---- Fig bars (per-family AUROC)
    fam = ["gull", "human", "heron"]
    o_mean = {f: np.mean([st[a][f] for a in OPUS]) for f in fam}
    o_sd = {f: np.std([st[a][f] for a in OPUS]) for f in fam}
    s_mean = {f: np.mean([st[a][f] for a in SONNET]) for f in fam}
    s_sd = {f: np.std([st[a][f] for a in SONNET]) for f in fam}
    bars = [("Textbook", {f: st["TB"][f] for f in fam}, None, "white", "///"),
            ("Karlsson", {f: st["K"][f] for f in fam}, None, "white", "\\\\\\"),
            ("Hand-fitted", {f: st["NB"][f] for f in fam}, None, "white", "..."),
            ("LLM Sonnet (4)", s_mean, s_sd, "0.75", ""),
            ("LLM Opus (5)", o_mean, o_sd, "0.25", ""),
            ("Measured positives*", {f: w17["realpos"][f]["auroc"] for f in fam}, None, "white", "xx"),
            ("Supervised*", {f: w17["sup"][f]["auroc"] for f in fam}, None, "white", "--")]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    wdt = 0.11
    for k, (lab, v, sd, col, h) in enumerate(bars):
        xs = np.arange(3) + (k - 3) * wdt
        ax.bar(xs, [v[f] for f in fam], wdt, yerr=[sd[f] for f in fam] if sd else None, color=col, hatch=h,
               edgecolor="k", lw=0.6, label=lab, capsize=2, error_kw={"lw": 0.6})
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Gulls", "Humans", "Heron (unseen by all)"])
    ax.set_ylabel("Track-level AUROC")
    ax.set_ylim(0.3, 1.03)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=6.5, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig_bars.png", dpi=600)
    fig.savefig(FIG / "fig_bars.pdf")
    plt.close(fig)
    heron_run = heron_timeline()
    # ---- tables
    ds = list(csv.DictReader((FIG / "table_dataset.csv").open()))
    order = ["D1", "D2", "D3", "D4", "D5", "D6", "seagull", "black-headed gull", "seagull and black-headed gull", "heron",
             "pigeon", "raven", "human_walk", "human_run", "CR"]
    pretty = {"seagull": "Gull", "black-headed gull": "Black-headed gull", "seagull and black-headed gull": "Mixed gulls",
              "heron": "Grey heron", "pigeon": "Pigeon", "raven": "Raven", "human_walk": "Human, walking",
              "human_run": "Human, running", "CR": "Corner reflector"}
    dsd = {r["class"]: r for r in ds}
    t1 = md_table("Classes of the SAAB SIRS recordings used in this study (dwells after removal of flagged edge cases).",
                  ["Class", "Recordings", "Dwells", "Duration (s)"],
                  [[pretty.get(c, "Drone " + c), dsd[c]["recordings"], dsd[c]["dwells"], f"{float(dsd[c]['duration_s']):.0f}"] for c in order],
                  "tbl:dataset")
    t2 = md_table("Simulators compared in this study. F is the spectral fitness on three fresh seeds (mean, dB). Run names: b = blind task, i = informed task, final digit = repetition.",
                  ["Name", "Agent", "Task", "Dominant moving-part model", "F (dB)"],
                  [[NAME[a], AGENT[a][0], AGENT[a][1], DESC[a], f"{F[a]:.2f}"] for a in HUMAN + OPUS + SONNET], "tbl:sims")
    rows = []
    for a in HUMAN + OPUS + SONNET:
        v = st[a]
        rows.append([NAME[a], f"{F[a]:.2f}", f"{v['closed']:.3f}", f"{v['gull']:.3f}", f"{v['human']:.3f}", f"{v['heron']:.3f}",
                     f"**{v['open']:.3f}**" if a in OPUS else f"{v['open']:.3f}", f"{v['D2_D6']:.2f}", f"{v['D1']:.2f}", f"{v['heron_fa']:.2f}"])
    for k, lab in (("realpos", "Measured positives*"), ("sup", "Supervised*")):
        v = w17[k]
        rows.append([lab, "—", "—", f"{v['gull']['auroc']:.3f}", f"{v['human']['auroc']:.3f}", f"{v['heron']['auroc']:.3f}",
                     f"{v['open_auroc_mean']:.3f}", f"{np.mean([v['detect_thr0'][c] for c in ['D2', 'D3', 'D4', 'D5', 'D6']]):.2f}",
                     f"{v['detect_thr0']['D1']:.2f}", f"{v['heron']['fa_thr0']:.2f}"])
    t3 = md_table("Detection on measured tracks (mean over three training seeds; track = 20 consecutive dwells). Open = mean AUROC over gulls, humans and the heron. Det. and FA are the drone detection rate and the heron false-alarm rate at logit zero. *Trained on measured data with recording-grouped five-fold cross-validation (one seed); the supervised detector saw gulls and humans in training.",
                  ["Simulator", "F (dB)", "In-sim AUROC", "Gulls", "Humans", "Heron", "Open", "Det. D2–D6", "Det. D1", "FA heron"],
                  rows, "tbl:main")
    ed = {"E1": "O", "E2": "S", "E3": "S2", "E4": "S4"}
    what = {"E1": "add the drifting point of S-b1", "E2": "remove the drift term", "E3": "remove the offset tone",
            "E4": "remove the −800 Hz shift"}
    t4 = md_table("Component edits. Each edited program was retrained with the same chain and seeds.",
                  ["Edit", "Change", "F parent → edited (dB)", "Open AUROC parent → edited", "Det. D2–D6 edited"],
                  [[NAME[e], what[e], f"{F[p]:.2f} → {F[e]:.2f}", f"{open_[p]:.3f} → {open_[e]:.3f}", f"{st[e]['D2_D6']:.2f}"]
                   for e, p in ed.items()], "tbl:edits")
    o_open = [open_[a] for a in OPUS]
    o_fit = [F[a] for a in ["O2", "O", "O3", "O4"]]
    num = {"llm_open": f"{np.mean(o_open):.3f}", "tb_open": f"{open_['TB']:.3f}", "k_open": f"{open_['K']:.3f}",
           "nb_open": f"{open_['NB']:.3f}", "n_sims": str(len(sims)), "o_group_fit_range": f"{min(o_fit):.2f}–{max(o_fit):.2f}",
           "tb_fit": f"{F['TB']:.2f}", "realpos_open": f"{w17['realpos']['open_auroc_mean']:.3f}",
           "sup_open": f"{w17['sup']['open_auroc_mean']:.4f}",
           "rho_fine": f"{rho_f:.2f}".replace("-", "−"), "p_fine": f"{pv_f:.2f}", "n_fine": str(len(fine)),
           "e1_fit": f"{F['E1']:.2f}", "e2_fit": f"{F['E2']:.2f}", "e2_open": f"{open_['E2']:.3f}",
           "nb_fit": f"{F['NB']:.2f}", "k_fit": f"{F['K']:.2f}", "k_d26": f"{st['K']['D2_D6']:.2f}",
           "k_human": f"{st['K']['human']:.3f}", "nb_d26": f"{st['NB']['D2_D6']:.2f}",
           **{f"thr_{k}{n}": f"{json.loads((FIG / 'heron_threshold.json').read_text())['thresholds'][th][m]:.3f}"
              for n, th in (("0", "0.0"), ("15", "1.5")) for k, m in (("fa", "heron_fa"), ("det", "det"))},
           "llm_minus_realpos": f"{np.mean(o_open) - w17['realpos']['open_auroc_mean']:.3f}", "sup_heron": f"{w17['sup']['heron']['auroc']:.3f}",
           "closed_min": f"{min(closed.values()):.3f}", "closed_max": f"{max(closed.values()):.3f}",
           "open_min": f"{min(open_.values()):.3f}", "open_max": f"{max(open_.values()):.3f}",
           "rho_fit": f"{rho:.2f}".replace("-", "−"), "p_fit": f"{pv:.3f}", "s_fit": f"{F['S']:.2f}", "d_fit": f"{F['D']:.2f}",
           "o_group_open_min": f"{min(o_open):.3f}", "o_group_open_max": f"{max(o_open):.3f}",
           "o_group_d26": f"{np.mean([st[a]['D2_D6'] for a in OPUS]):.2f}",
           "o_heron_fa": f"{np.mean([st[a]['heron_fa'] for a in OPUS]):.2f}", "o_heron": f"{np.mean([st[a]['heron'] for a in OPUS]):.3f}",
           "e1_open": f"{open_['E1']:.3f}", "o_open": f"{open_['O']:.3f}",
           "e2_drop": f"{max(open_[p] - open_[e] for e, p in ed.items() if e != 'E1'):.2f}",
           "table_dataset": t1, "table_sims": t2, "table_main": t3, "table_edits": t4,
           "fig_bars": "![Track-level AUROC per non-drone family. Bars for the LLM groups show the mean over simulators with the standard deviation across simulators; other bars are single simulators or references. *Trained on measured data.](figs/fig_bars.png){#fig:bars width=95%}"}
    src = (P / "manuscript_template.md").read_text(encoding="utf8")
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", src)) - set(num))
    for k, v in num.items():
        src = src.replace("{{" + k + "}}", v)
    # ---- numeric citations in order of first appearance (MDPI style)
    bib = (P / "references.bib").read_text(encoding="utf8")
    entries = {m.group(1): m.group(0) for m in re.finditer(r"@\w+\{([^,]+),.*?\n\}", bib, re.S)}
    order, unknown = [], set()

    def cite(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in entries:
                unknown.add(k)
                nums.append("?")
                continue
            if k not in order:
                order.append(k)
            nums.append(str(order.index(k) + 1))
        return "[" + ",".join(nums) + "]"
    body_docx = re.sub(r"\[(@[^\]]+)\]", cite, src)
    refs = []
    for i, k in enumerate(order, 1):
        refs.append(f"{i}. {fmt_ref(entries[k])}")
    body_docx += "\n" + "\n".join(refs) + "\n"
    body_docx = re.sub(r"^bibliography:.*\n", "", body_docx, flags=re.M)
    (P / "manuscript.md").write_text(body_docx, encoding="utf8")
    subprocess.run([PANDOC, str(P / "manuscript.md"), "-o", str(P / "Manuscript_RS_draft.docx"), "--resource-path", str(P)], check=True)
    # LaTeX body for the MDPI Overleaf template
    tex_src = re.sub(r"\[(@[^\]]+)\]", lambda m: "\\cite{" + ",".join(k.strip().lstrip("@") for k in m.group(1).split(";")) + "}", src)
    tex_src = re.sub(r"^bibliography:.*\n", "", tex_src, flags=re.M)
    (P / "manuscript_for_latex.md").write_text(tex_src, encoding="utf8")
    subprocess.run([PANDOC, str(P / "manuscript_for_latex.md"), "-o", str(P / "body.tex"), "--wrap=preserve"], check=True)
    # ---- supplementary material
    task = (W9 / "opus_blind_r2" / "TASK.md").read_text(encoding="utf8")
    task = re.sub(r"[A-Z]:\\research\\openset_physics\\[^\s`\"]*", "<path>", task)
    sc = json.loads((W9 / "score.json").read_text())
    runmap = {"O2": "opus_blind_r1", "O": "opus_blind_r2", "O3": "opus_informed_r1", "O4": "opus_informed_r2",
              "S": "sonnet_blind_r1", "S2": "sonnet_blind_r2", "S3": "sonnet_informed_r1", "S4": "sonnet_informed_r2"}
    tf = md_table("Spectral fitness F (dB) of the best program of each agent run on three fresh seeds, and the number of submissions used.",
                  ["Program", "Seed 101", "Seed 202", "Seed 303", "Mean", "Submissions"],
                  [[NAME[a]] + [f"{x:.2f}" for x in sc[d]["fresh_fitness"]] + [f"{sc[d]['fresh_mean']:.2f}", sc[d]["n_submissions"]]
                   for a, d in runmap.items()], "tbl:s1")
    ts = md_table("Open-set AUROC of every simulator for each training seed.",
                  ["Simulator", "Seed 31", "Seed 32", "Seed 33", "Mean"],
                  [[NAME[a]] + [f"{x:.3f}" for x in st[a]["per_seed_open"]] + [f"{st[a]['open']:.3f}"] for a in sims], "tbl:s2")
    w14 = json.loads((R / "W14_summary.json").read_text())
    w14b = json.loads((R / "W14b_summary.json").read_text())
    sup = (P / "supplementary_template.md").read_text(encoding="utf8")
    for k, v in {"task_blind": task.strip(), "table_fit_seeds": tf, "table_seeds": ts,
                 "h1_diff": f"{w14['H1']['O_minus_S']:.3f}", "h1_seeds": ", ".join(f"+{x:.3f}" for x in w14["H1"]["per_seed"]),
                 "w14b_diff": f"+{w14b['primary']['diff']:.3f}", "w14b_wins": str(w14b["primary"]["pair_wins_of_16"])}.items():
        sup = sup.replace("{{" + k + "}}", v)
    missing += sorted(set(re.findall(r"\{\{(\w+)\}\}", sup)))
    (P / "supplementary.md").write_text(sup, encoding="utf8")
    subprocess.run([PANDOC, str(P / "supplementary.md"), "-o", str(P / "Supplementary_RS_draft.docx")], check=True)
    meta = {"numbers": {k: v for k, v in num.items() if not k.startswith(("table", "fig"))}, "missing_placeholders": missing,
            "unknown_citations": sorted(unknown), "n_references": len(order), "heron_timeline_net": heron_run,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (P / "build_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: meta[k] for k in ("missing_placeholders", "unknown_citations", "n_references")}, indent=1))
    print(json.dumps(meta["numbers"], indent=1, ensure_ascii=False))


def fmt_ref(e):
    """Minimal MDPI-like formatting from a BibTeX entry."""
    g = lambda f: (re.search(rf"\b{f}\s*=\s*[{{\"](.+?)[}}\"]\s*,?\s*\n", e, re.S) or [None, ""])[1].replace("{", "").replace("}", "").strip()
    e = (e.replace("\\'e", "é").replace("\\'a", "á").replace("\\\"a", "ä").replace("\\\"o", "ö").replace("\\,", " ")
         .replace("\\&", "&").replace("--", "–"))
    au = g("author")
    names = []
    for a in [x.strip() for x in au.split(" and ") if x.strip()]:
        if "," in a:
            last, first = [s.strip() for s in a.split(",", 1)]
        else:
            parts = a.split()
            last, first = parts[-1], " ".join(parts[:-1])
        ini = " ".join(p[0] + "." for p in re.split(r"[\s\-]+", first) if p)
        names.append(f"{last}, {ini}".strip(", "))
    au_s = "; ".join(names[:10]) + ("; et al." if len(names) > 10 else "")
    title, year = g("title"), g("year")
    venue = g("journal") or g("booktitle") or g("school") or g("publisher") or g("howpublished")
    vol, num, pages, doi = g("volume"), g("number"), g("pages"), g("doi")
    s = f"{au_s} {title}. *{venue}* **{year}**"
    if vol:
        s += f", *{vol}*"
    if pages:
        s += f", {pages.replace('--', '–')}"
    s += "."
    if doi:
        s += f" https://doi.org/{doi}."
    elif g("url"):
        s += f" {g('url')}."
    return s


if __name__ == "__main__":
    main()
