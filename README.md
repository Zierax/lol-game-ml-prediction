# LoL 10-Minute Match Outcome Prediction: A Mutual-Information Audit

Code and committed artifacts behind the manuscript *A Mutual-Information Audit
of Early-Game League of Legends Match Outcome Prediction: What Eight Calibrated
Scores Achieve at Ten Minutes* (`paper/entropy_submission/`).

## What this repository actually shows

Eight model families (LR, RF, XGBoost, LightGBM, RBF-SVM, k-NN, MLP,
soft-voting) converge to 0.69–0.72 test accuracy. The manuscript asks what that
plateau does and does not license, and the audit is deliberately narrow about
its own instrument:

- **We measure the information in a fitted score, never in the snapshot.** The
  quantity estimated is $I(\hat S; Y)$, and the data-processing inequality makes
  it a lower bound on $I(X; Y)$. A crippled score would return the same low
  reading as an information-poor snapshot, so **a low measurement cannot refute
  an information ceiling**. It can only fail to detect one.
- **It does fix a budget.** Fano's inequality is tight for a binary target, so
  an accuracy target implies a minimum information requirement: 80% needs
  0.278 bits, 90% needs 0.531 bits, against the 0.230 bits the best of the eight
  scores carries.
- **The evidence against the ceiling reading comes from elsewhere:** the
  learning curves read on the information axis (logistic regression's achieved
  information is flat from 40% of the data onward, curvature $p = 0.0009$,
  Bonferroni-robust; LightGBM is still rising), the 54-fold spread of minute-10
  information across final-duration strata, and the label-noise audit.

Consequences for how to read the tables: $A_{\max}(\hat I)$ is a **point
estimate** of what these eight score functions can attain — not a certified
upper bound on them, and not a bound on the task.

## Setup

```bash
pip install -r requirements.txt
```

Determinism: all estimators are seeded (seed 42) and single-threaded, so
repeated runs are bit-identical. `GridSearchCV` parallelism is controlled by
the `SEARCH_JOBS` environment variable (default `1` = fully deterministic,
single-threaded search; set `SEARCH_JOBS=-1` to run CV folds in parallel for
speed without changing results).

## Reproducing the paper

Run these from the repository root, in this order. Together they regenerate
every artifact and every figure in the manuscript.

| Command | Artifacts produced | Manuscript items |
|---|---|---|
| `python run_ceiling.py` | `plots_ceiling/ceiling_summary.json`, `fano_curve.png`, `mi_spectrum.png`, `pca_variance.png`, `learning_curve.png` | Table 1 (8-family zoo), Table 3 (binned vs binning-free MI), Table 4 (McNemar), Table 5 (learning curves incl. MI axis), Figures 1–4 |
| `python run_bounds_audit.py` | `plots_ceiling/bounds_audit.json` | Score-Bayes ceiling and the binning-free Ross (2014) estimator; backs the estimator comparison |
| `python run_label_noise_audit.py` | `plots_ceiling/label_noise_audit.json`, `label_noise_sweep.png` | Table 6 (label-noise injection, $\rho \in \{0, 0.1, 0.2, 0.3\}$, $t$-intervals over repeats) |
| `python run_temporal_ceiling.py` | `plots_temporal/temporal_summary.{csv,json}`, `temporal_fixed_cohort.{csv,json}`, `temporal_lead_padded.{csv,json}`, `temporal_duration_stratified.{csv,json}`, `temporal_ceiling.png` | Table 7 (duration strata), the fixed-cohort and lead-padded controls, Figure 5 |
| `python run_temporal_accrual.py` | `plots_temporal/temporal_accrual.json` | Information-accrual rate (0.0173 ± 0.0009 bits/min, $R^2 = 0.982$) |
| `python redraw_paper_figures.py` | `paper/entropy_submission/figures/*.png` at 600 dpi | The seven submission figures |

Macro-averaged McNemar over the 21 model pairs yields 9 nominally significant
pairs; 5 survive Holm correction across those 21 pairs, and Table 4 reports the
7 rows with $p < 0.05$ the paper has room for plus a note on the two omitted.

### Verification scripts

These check the manuscript against the committed JSON, and are what the authors
ran before submission. They are not required to reproduce the results.

| Command | What it checks |
|---|---|
| `python verify_paper.py` | Every table cell, range, confidence interval, $p$-value and framing claim against the artifacts (prints `ALL CHECKS PASSED`) |
| `python verify_references.py` | Resolves every DOI in the bibliography through Crossref → `artifacts/reference_check.json` |
| `python verify_rank_correlations.py` | Spearman(accuracy, MI) = 0.205 and Spearman(AUC, MI) = 0.976 across the eight families |
| `python verify_submission.py` | Submission ZIP ↔ working tree hash match, required-file and build-junk check |

## Layout

- `paper/entropy_submission/` — manuscript source, compiled PDF, 600 dpi
  figures, and `artifacts/` (the JSON backing every reported number).
- `src/` — shared split, model zoo, calibration and MI-estimation code.
- `plots_ceiling/`, `plots_temporal/` — analysis outputs at the repository level.

Legacy from the earlier conference submission and not used by the manuscript:
`main.py`, `main_v2.py`, `redraw_plots.py`.

## Data

- `data/high_diamond_ranked_10min.csv` (9,879 matches; 40 raw columns, target
  `blueWins`). After removing the identifier and three perfectly redundant
  mirrored columns (`redGoldDiff`, `redExperienceDiff`, `redFirstBlood`), 35 raw
  features; engineering yields an 87-column matrix.
- `data/lol_ranked_games_temporal.csv` (242,572 rows, 24,912 games, per
  (game, minute) frames from minute 10 every 2 minutes; 55 per-frame features).

Both datasets are public on Kaggle (uploader bobbyscience); full citations are in
`paper/entropy_submission/main.tex`.

## License

MIT — see `LICENSE`.
