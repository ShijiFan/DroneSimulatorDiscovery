"""v6 checks (2026-11-04) -> data/v6_checks.json, read by build_article_v6.py.
taper: measured power-envelope FWHM of D1-D6 dwells; fitness with 9 vs 5 ms taper (taper_fit.json, from taper_fit.py);
       open-set AUROC of W14 runs with the 9 ms taper (W14_<arm>_s3x) and the 5 ms taper (W14_<arm>_tap5_s3x, taper_train.py).
pfa_pooled: false-alarm rate at logit zero on pooled gull+human+heron tracks, mean over seeds (W14 result.json, W17_summary.json).
tracks: evaluation tracks per class (20 retained dwells, as in e3w8_sim_rotor_detector.load_real)."""
import glob, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path(r'E:\research\openset_physics\runs')
sys.path.insert(0, r'E:\research\openset_physics\e0_identifiability')
from e3w8_sim_rotor_detector import DRONES, TRACK, load_real  # noqa: E402

OPEN = ['gull', 'human', 'heron']


def runs(arm):
    out = []
    for d in sorted(RUNS.glob(f'W14_{arm}_s3[123]_*')):
        if (d / 'result.json').exists():
            out.append(json.loads((d / 'result.json').read_text())['track20'])
    return out


def pooled_fa(r):
    return sum(r[f]['fa_thr0'] * r[f]['n_neg'] for f in OPEN) / sum(r[f]['n_neg'] for f in OPEN)


# measured illumination
d = np.load(r'E:\rag\datasets\data3\data_SAAB_SIRS_77GHz_FMCW.npy', allow_pickle=True)
lab = np.array([r[0][0] for r in d])
fw = {}
for c in DRONES:
    E = []
    for i in np.flatnonzero(lab == c):
        p = np.abs(d[i, 1][512:768]) ** 2
        E.append(p / p.mean(0, keepdims=True))
    e = np.concatenate(E, 1).mean(1)
    idx = np.flatnonzero(e >= e.max() / 2)
    fw[c] = (idx[-1] - idx[0] + 1) / 17e3 * 1e3

F = json.loads((HERE / 'taper_fit.json').read_text())
opus = ['O-b1', 'O-b2', 'O-i1', 'O-i2', 'O-b1-dist']
auc9 = {a: float(np.mean([r['open_auroc_mean'] for r in runs(a)])) for a in ('D', 'NB', 'K', 'TB')}
auc5 = {}
for a in ('D', 'NB', 'K', 'TB'):
    r = runs(a + '_tap5')
    assert len(r) == 3, (a, len(r))
    auc5[a] = float(np.mean([x['open_auroc_mean'] for x in r]))
taper = dict(meas_pfwhm=fw, meas_pfwhm_lo=min(fw.values()), meas_pfwhm_hi=max(fw.values()),
             F9={k: v['0.009'] for k, v in F.items()}, F5={k: v['0.005'] for k, v in F.items()},
             F_opus_dmin=min(F[k]['0.005'] - F[k]['0.009'] for k in opus), F_opus_dmax=max(F[k]['0.005'] - F[k]['0.009'] for k in opus),
             auc9=auc9, auc5=auc5)

pfa = {a: float(np.mean([pooled_fa(r) for r in runs(a)])) for a in ('TB', 'K', 'Km', 'NB', 'O', 'O2', 'O3', 'O4', 'D', 'S', 'S2', 'S3', 'S4')}
W17 = json.loads(Path(r'E:\research\openset_physics\paper_RS\FINAL_SUBMISSION_RS_20261015\04_Code_and_Programs\W17_summary.json').read_text())
nn = runs('D')[0]
for k in ('realpos', 'sup'):
    pfa[k] = sum(W17[k][f]['fa_thr0'] * nn[f]['n_neg'] for f in OPEN) / sum(nn[f]['n_neg'] for f in OPEN)

n = {}
for lb, i, X in load_real():
    n[lb] = n.get(lb, 0) + len(X) // TRACK
per = [n[c] for c in DRONES]
tracks = dict(drone=sum(per), drone_min=min(per), drone_max=max(per), gull=nn['gull']['n_neg'], human=nn['human']['n_neg'],
              heron=nn['heron']['n_neg'], CR=nn['CR']['n_neg'], per_class=n)
assert tracks['gull'] == n['seagull'] + n['black-headed gull'] + n['seagull and black-headed gull']

agent8 = float(np.mean([np.mean([r['open_auroc_mean'] for r in runs(a)]) for a in ('O', 'O2', 'O3', 'O4', 'S', 'S2', 'S3', 'S4')]))
opus5 = float(np.mean([np.mean([r['open_auroc_mean'] for r in runs(a)]) for a in ('O', 'O2', 'O3', 'O4', 'D')]))
out = dict(taper=taper, pfa_pooled=pfa, realpos_fa={f: W17['realpos'][f]['fa_thr0'] for f in OPEN}, tracks=tracks,
           mean_agent8=agent8, mean_opus5=opus5)
(HERE.parent / 'v6_checks.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'tracks'}, indent=1), tracks['drone'], agent8, opus5)
