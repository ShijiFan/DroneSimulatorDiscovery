"""André TAES round 2: statistics for the 9 ms (paper) and 5 ms (measured) tapers with the W18 method
(track = 20 dwells; bootstrap over recordings within each class, B = 2000, seed 0; AUROC with ties counted 1/2).
Inputs: 9 ms dwell scores E:/research/openset_physics/runs/W18_scores (W14 runs) and W17 out-of-fold scores (E:);
        5 ms dwell scores ./runs/W18_scores (from w18_rescore.py run on ./runs) and ./runs/W17_realpos_* (F:).
Output: tap5_stats.json.  Adds the AUROC pooled over all gull, human and heron tracks (André) and P_d / P_fa at logit 0."""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, r'E:\research\openset_physics\e0_identifiability')
import w18_stats as W

HERE = Path(__file__).resolve().parent
E_RUNS = Path(r'E:\research\openset_physics\runs')
F_RUNS = HERE / 'runs'
ARMS = ["TB", "K", "Km", "NB", "O2", "O", "O3", "O4", "D", "S", "S2", "S3", "S4", "E1", "E2", "E3", "E4"]
NAME = dict(TB='Textbook', K='Karlsson', Km='K-m', NB='Hand-fitted', O2='O-b1', O='O-b2', O3='O-i1', O4='O-i2', D='O-b1-dist',
            S='S-b1', S2='S-b2', S3='S-i1', S4='S-i2', E1='O-b2 + drift', E2='S-b1 - drift', E3='S-b2 - tone', E4='S-i2 - shift',
            realpos='Measured positives')
FIT = json.loads(Path(r'E:\research\openset_physics\paper_RS\rs_v6_20261104\data\v6_checks\taper_fit.json').read_text())
FITKEY = dict(TB='Textbook', K='Karlsson', Km='K-m', NB='Hand-fitted', O2='O-b1', O='O-b2', O3='O-i1', O4='O-i2', D='O-b1-dist',
              S='S-b1', S2='S-b2', S3='S-i1', S4='S-i2', E1='O-b2+drift', E2='S-b1-drift', E3='S-b2-tone', E4='S-i2-shift')
OPEN_CLS = [c for f in W.OPEN for c in W.FAMS[f]]


def load(score_dir, suffix):
    by = {}
    for p in sorted(score_dir.glob('*.npz')):
        m = re.match(r'W14_(.+)_s(\d+)_\d{8}_\d{6}$', p.stem)
        if not m or not m[1].endswith(suffix):
            continue
        arm = m[1][:len(m[1]) - len(suffix)] if suffix else m[1]
        if arm not in ARMS:
            continue
        z = np.load(p)
        by.setdefault(arm, []).append((int(m[2]), p.stem, {int(i): (str(l), z[f'r{int(i):03d}']) for l, i in zip(z['labels'], z['index'])}))
    return by


def w17(runs_dir):
    rec = {}
    for p in sorted(runs_dir.glob('W17_realpos_f*_*/oof_scores.json')):
        seed = json.loads((p.parent / 'meta.json').read_text())['args']['seed']
        for key, s in json.loads(p.read_text()).items():
            i, lab = key.split('|', 1)
            rec.setdefault(seed, {})[int(i)] = (lab, np.asarray(s, float))
    return [(seed, f'W17_realpos_s{seed}', r) for seed, r in sorted(rec.items()) if len(r) > 100]


def extra(t, groups):
    """pooled AUROC over all open-set tracks, P_d (D1, D2-D6) and pooled P_fa at logit 0."""
    pos = np.concatenate([t[r] for c in W.DRONES for r in groups[c]])
    neg = np.concatenate([t[r] for c in OPEN_CLS for r in groups.get(c, [])])
    d1 = np.concatenate([t[r] for r in groups['D1']])
    d26 = np.concatenate([t[r] for c in W.DRONES[1:] for r in groups[c]])
    return dict(pooled=float(W.auc(pos, neg)), pd_D1=float((d1 > 0).mean()), pd_D26=float((d26 > 0).mean()), pfa=float((neg > 0).mean()))


def analyse(by, tag):
    any_rec = next(iter(by.values()))[0][2]
    groups = {}
    for i, (l, s) in any_rec.items():
        groups.setdefault(l, []).append(i)
    classes = [c for c in groups if c in W.DRONES or any(c in v for v in W.FAMS.values())]
    trk = {a: [(seed, {i: W.tracks(s) for i, (l, s) in rec.items()}) for seed, n, rec in lst] for a, lst in by.items()}
    point = {}
    for a, lst in trk.items():
        vals = [{**W.stats_for(t, groups), **extra(t, groups)} for seed, t in lst]
        point[a] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        point[a]['per_seed_open'] = {seed: float(W.stats_for(t, groups)['open']) for seed, t in lst}
    rng = np.random.default_rng(0)
    boot = {a: [] for a in trk}
    for b in range(W.B):
        g = {c: list(rng.choice(groups[c], len(groups[c]), replace=True)) for c in classes}
        for a, lst in trk.items():
            boot[a].append(float(np.mean([W.stats_for(t, g)['open'] for seed, t in lst])))
    ci = {a: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for a, v in boot.items()}
    diffs = {}
    opus = [a for a in ('O2', 'O', 'O3', 'O4', 'D') if a in boot]
    if len(opus) == 5:
        om = np.mean([boot[a] for a in opus], 0)
        for ref in ('NB', 'K', 'Km', 'TB', 'realpos'):
            if ref in boot:
                diffs[f'opus_mean_minus_{ref}'] = dict(point=float(np.mean([point[a]['open'] for a in opus]) - point[ref]['open']),
                                                       ci95=[float(np.percentile(om - boot[ref], q)) for q in (2.5, 97.5)])
    sims = [a for a in ARMS if a in point and a != 'Km']
    corr = {}
    if len(sims) == 16:
        fk = '0.009' if tag == '9ms' else '0.005'
        Fv = np.array([FIT[FITKEY[a]][fk] for a in sims]); Av = np.array([point[a]['open'] for a in sims])
        for cut in (None, 2.5, 2.4, 2.3):
            m = np.ones(len(sims), bool) if cut is None else Fv < cut
            r = spearmanr(Fv[m], Av[m])
            corr['all' if cut is None else f'F<{cut}'] = dict(n=int(m.sum()), rho=float(r.statistic), p=float(r.pvalue))
    return dict(point=point, ci95=ci, diffs=diffs, spearman_fit_vs_open=corr, seeds={a: len(l) for a, l in by.items()})


def main():
    out = {}
    b9 = load(E_RUNS / 'W18_scores', '')
    b9['realpos'] = w17(E_RUNS)
    out['9ms'] = analyse(b9, '9ms')
    b5 = load(F_RUNS / 'W18_scores', '_tap5')
    rp = w17(F_RUNS)
    if rp:
        b5['realpos'] = rp
    out['5ms'] = analyse(b5, '5ms')
    (HERE / 'tap5_stats.json').write_text(json.dumps(out, indent=1))
    for a in ARMS + ['realpos']:
        r9 = out['9ms']['point'].get(a); r5 = out['5ms']['point'].get(a)
        f = lambda r, c: f"{r['open']:.3f} [{c[0]:.3f},{c[1]:.3f}] pooled {r['pooled']:.3f} D1 {r['pd_D1']:.2f} D2-6 {r['pd_D26']:.2f} Pfa {r['pfa']:.2f}" if r else '-'
        print(f"{NAME[a]:18s} 9ms {f(r9, out['9ms']['ci95'].get(a))} | 5ms {f(r5, out['5ms']['ci95'].get(a))}")
    print(json.dumps({k: (out[k]['diffs'], out[k]['spearman_fit_vs_open']) for k in out}, indent=1))


if __name__ == '__main__':
    main()
