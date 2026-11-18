"""André TAES review, round 2 (2026-11-16): retrain detectors with the measured beam taper (5 ms amplitude FWHM;
measured power-envelope FWHM of D1-D6 dwells is 3.4-3.7 ms, i.e. about 5 ms in amplitude), positives and negatives alike.
Same change as taper_train_20261104_original.py (used for D/NB/K/TB on 2026-11-04); the only other difference is that
outputs go to ./runs on F: (E: is full). The training scripts are imported unchanged from E:; the copies in
./e0_identifiability are byte-identical and serve only as __file__ anchors (output folder and script hashes).
Usage: python tap5_train.py w14 --arm O_tap5 --bank ... --seed 31
       python tap5_train.py w17 --mode realpos --fold 0 --seed 31"""
import sys
from pathlib import Path

E0 = r'E:\research\openset_physics\e0_identifiability'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, E0)
import sim_saab
sim_saab.FWHM = 5e-3
which = sys.argv.pop(1)
if which == 'w14':
    import w14_train as m
    anchor = 'w14_train.py'
else:
    import w17_real_ref as m
    anchor = 'w17_real_ref.py'
import e3w8b_ablate
assert e3w8b_ablate.S.FWHM == 5e-3 and sim_saab.FWHM == 5e-3
for p in ['w14_train.py', 'e3w8b_ablate.py', 'e3w8_sim_rotor_detector.py', 'sim_saab.py', 'train_e0b.py', 'w17_real_ref.py']:
    assert (HERE / 'e0_identifiability' / p).read_bytes() == (Path(E0) / p).read_bytes(), p
m.__file__ = str(HERE / 'e0_identifiability' / anchor)
m.main()
