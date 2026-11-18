# Drone Simulator Discovery

Research software and evaluation results for *Simulator Discovery With Language-Model Agents for Open-Set Drone Detection in Millimeter-Wave Surveillance Radar* by Shiji Fan, Daniel Albuquerque, André Rouco, and Ming Shen.

The study scores candidate rotor simulators against a measured mean residual Doppler spectrum. The detectors are trained on simulated samples and evaluated on measured tracks. The repository contains the simulator programs, experimental scripts, 9 ms result summaries, and the 5 ms sensitivity analysis based on 13 simulators and three training seeds.

## Reproduce the 5 ms analysis

From the repository root, run:

    python -m pip install -r requirements.txt
    cd taper_sensitivity_20261116
    python tap5_stats.py --five-ms-only

This recomputes the point estimates and 2,000 recording-level bootstrap resamples from the 39 archived sets of dwell scores. Compare the generated tap5_stats_recomputed_5ms.json with the 5ms section of tap5_stats.json. The per-seed open-set AUROCs are also available in per_seed_5ms.csv. The release was tested with Python 3.14.2, NumPy 2.4.1, and SciPy 1.17.0.

## Repository contents

| Path | Contents |
| --- | --- |
| simulators/ | Rotor simulator programs, hand-designed baselines, edits, and ablations |
| agent_tasks/ | Task descriptions used in the program searches |
| scripts/ | Original simulation, training, evaluation, and plotting scripts |
| W*_summary*.json and W18_tables.json | Main 9 ms results and analysis summaries |
| taper_sensitivity_20261104/ | Preliminary beam-taper checks |
| taper_sensitivity_20261116/ | Final 5 ms scripts, 39 result JSON files, 39 score archives, and frozen aggregates |

The original training and plotting scripts preserve their research workstation paths and require configuration for rerunning on another machine. Full retraining requires the public radar dataset, simulation banks, PyTorch and scikit-learn.

## Data and licensing

The measurements are from Alexander Karlsson's *Radar measurements on drones, birds and humans with a 77GHz FMCW sensor*, [Zenodo](https://doi.org/10.5281/zenodo.5511911), using the v2.0 NumPy files, licensed CC BY 4.0 by the data provider. The score archives contain recording indices, class labels, and model scores derived from these measurements. Obtain the raw radar recordings from Zenodo.

The original source code is released under the MIT License. See [LICENSE](LICENSE). Use [CITATION.cff](CITATION.cff) when citing this software.
