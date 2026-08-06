# LoL 10-Minute Match Outcome Prediction: The Information Ceiling

Machine-learning pipeline that predicts the winner of League of Legends matches
from the first 10 minutes of gameplay, and an information-theoretic analysis
showing the celebrated ~72% accuracy plateau is an *information ceiling*, not a
model-deficiency artifact.

## Project Overview

The task is a binary classification of whether the blue team wins based on
early-game features at the 10-minute mark. Eight model families (LR, RF,
XGBoost, LightGBM, RBF-SVM, k-NN, MLP, soft-voting) converge to 0.69-0.72
accuracy. Using mutual-information estimation and Fano's inequality, we show
the 10-minute snapshot carries at most ~0.24 bits of decision-relevant
information, capping achievable accuracy near 0.72-0.78. A second, independent
per-frame dataset (24,912 matches) reproduces the plateau and shows the ceiling
rises with the observation window (72.9% -> 84.0% from minutes 10-30), with
fixed-cohort / lead-padded / label-noise controls ruling out composition and
label-noise explanations of the temporal signal.

## Setup

```bash
pip install -r requirements.txt
```

Determinism: all estimators are seeded (seed 42) and single-threaded, so
repeated runs are bit-identical. `GridSearchCV` parallelism is controlled by
the `SEARCH_JOBS` environment variable (default `1` = fully deterministic,
single-threaded search; set `SEARCH_JOBS=-1` to run CV folds in parallel for
speed without changing results).

## Reproducing the paper's claims

| Command | Artifacts produced | Claims it supports |
|---------|--------------------|--------------------|
| `python run_ceiling.py` | `plots_ceiling/ceiling_summary.json`, fano_curve.png, mi_spectrum.png, pca_variance.png, learning_curve.png | Table 1 (8-family zoo), MI spectrum, PCA redundancy, learning-curve saturation, 21-pair McNemar (2 significant after Holm), Figure "fano_curve" |
| `python run_temporal_ceiling.py` | `plots_temporal/temporal_summary.{csv,json}`, temporal_fixed_cohort.{csv,json}, temporal_lead_padded.{csv,json}, temporal_duration_stratified.{csv,json}, temporal_ceiling.png | Temporal rise (72.9% -> 84.0%), fixed-cohort controls, lead-padded control, composition table (minute-10 accuracy by final duration) |
| `python run_label_noise_audit.py` | `plots_ceiling/label_noise_audit.json`, label_noise_sweep.png | Label-noise audit (rho in {0 .. 0.20}); rules out label noise as the cause of the information-poor long-game subset |
| `python main_v2.py` | `plots_v2/model_metrics_v2.json`, calibration_results.json, threshold_oos.json | Leakage-free (train/val/test) tuning, validation-based threshold optimisation (F1 0.725->0.743), out-of-sample calibration |
| `python main.py` | `plots/model_metrics.json`, nested_cv_results.json, confidence_intervals.json, threshold_optimisation.json, pairwise_significance.json | Legacy v1 pipeline with validation-based threshold optimisation (no test-set double-dipping) |

The paper source is `paper/information_ceiling/main.tex` (compiled with
`pdflatex` + `bibtex`). Figures are loaded from `plots_ceiling/` and
`plots_temporal/` via the graphics path in the preamble.

## Data

- `data/high_diamond_ranked_10min.csv` (9,879 matches, ascending: 40 raw
  columns; target `blueWins`). After removing the identifier and three
  perfectly redundant mirrored columns (`redGoldDiff`, `redExperienceDiff`,
  `redFirstBlood`), 35 raw features; engineering yields an 87-column matrix.
- `data/lol_ranked_games_temporal.csv` (242,572 rows, 24,912 games, per
  (game, minute) frames from minute 10 every 2 minutes; 55 per-frame features).

Both datasets are public on Kaggle (uploader bobbyscience; see
`paper/information_ceiling/references.bib`).