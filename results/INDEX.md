# Where each result comes from

All files are written by `./run_all.sh` (run from the repo root). Script → output.

| Step | Script | Output file | What it contains |
|---|---|---|---|
| 1. Select and clean curves | `src/prep.py` | `results/prep_output.txt` | 44 curves / 12 papers after curation filters and dosing-schedule parsing; counts per analyte and dosing kind |
| 2. Per-curve Bateman fits | `src/model.py` | `results/fit_output.txt` | Fitted V, ka, ke, t½ and in-sample log10 RMSE per curve (median 0.12); 42 curves with ≥4 points |
| 3. Leave-one-paper-out (main result) | `src/lopo.py` | `results/lopo_output.txt` | Paper-level MAE for 5 models with bootstrap 95% intervals; subsets (yes+caution only, tenofovir only, analyte seen); per-paper GBM / mech_pop / hybrid MAE |
| | | `results/lopo_per_curve_results.csv` | Same, one row per curve |
| 4. Dose and population extrapolation | `src/extrap.py` | `results/extrap_output.txt`, `results/extrap_per_curve.csv` | Leave-one-dose-out (PMID 11557462) and leave-one-population-out (pregnancy, renal, DDI) |
| 5A. Dose-scaled control | `src/extras.py A` | `results/extras_AB_output.txt` (section A), `results/dose_control_per_curve.csv` | GBM on log10(C/dose) vs the other models on held-out doses |
| 5B. Sensitivity to PMID 33252495 | `src/extras.py B` | `results/extras_AB_output.txt` (section B), `results/lopo_full_extras.csv`, `results/lopo_without_33252495_extras.csv` | LOPO with it in/out; points per curve by paper |
| 5C. Learning curve | `src/extras.py C` | `results/learning_curve_output.txt`, `results/learning_curve.png`, `results/learning_curve_summary.csv`, `results/learning_curve_per_curve.csv` | MAE vs number of training papers (k = 2, 4, 6, 8, 11) |
| 5C′. Learning curve without PMID 33252495 | `src/extras.py C2` | `results/learning_curve_output_excl33252495.txt`, `results/learning_curve_excl33252495.png`, `results/learning_curve_summary_excl33252495.csv`, `results/learning_curve_per_curve_excl33252495.csv` | Same as 5C with that paper removed from training pool and test set (k = 2, 4, 6, 8, 10) |

Inputs: `data/curves_long.csv` (PEAK points), `data/curve_metadata.csv` (hand curation of 193 curves).

Note: `results/extras_AB_output.txt` and the `*_extras.csv` files use the same LOPO folds as `lopo.py` but add the GBM-on-C/dose models, so mean/GBM/hybrid numbers match `lopo_output.txt`.
