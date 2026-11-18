# 5 ms beam-taper sensitivity

Supplementary Section S7 and Table S4 compare the 5 ms amplitude taper with the main 9 ms condition. This directory contains 39 recorded training summaries and 39 sets of dwell scores for the 13 simulators at seeds 31, 32 and 33. It includes the original launcher, its dependencies as archived, the frozen aggregate and the per-seed open-set AUROC table. Checkpoints and raw radar recordings are omitted from the reviewer archive.

Run the following command from the directory containing this README after extracting the whole reviewer archive:

    python tap5_stats.py --five-ms-only

NumPy and SciPy are sufficient. The script writes tap5_stats_recomputed_5ms.json. The archived result tap5_stats.json is preserved. The score files are grouped by recording. The analysis uses 20-dwell tracks and 2,000 recording-level bootstrap resamples with seed 0. The 9 ms aggregates are also in the frozen JSON. A full recomputation requires the original 9 ms scores, which can be located using TAES_9MS_RUNS.

tap5_stats_as_run.py preserves the original aggregation as executed on the authors' workstation. tap5_train.py and its training dependencies retain workstation paths and require the measured data and simulation banks. The four simulator arms initially run on 4 November contribute 12 of the 39 runs. The remaining nine arms contribute 27 runs from 16 November.
