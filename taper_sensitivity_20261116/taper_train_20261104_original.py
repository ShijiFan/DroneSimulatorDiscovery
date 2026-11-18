"""André RS review check (2026-11-04): retrain W14 detectors with the measured beam taper. The only change from w14_train.py
is sim_saab.FWHM = 5 ms (amplitude; measured mean power-envelope FWHM of D1-D6 dwells is 3.4-3.7 ms, i.e. ~5 ms in amplitude),
applied to positives and negatives alike. Usage: python taper_train.py --arm D_tap5 --bank ... --seed 31"""
import sys
sys.path.insert(0, r'E:\research\openset_physics\e0_identifiability')
import sim_saab
sim_saab.FWHM = 5e-3
import w14_train
assert w14_train.S.FWHM == 5e-3
w14_train.main()
