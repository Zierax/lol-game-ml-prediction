# Project Title
## LoL 10‑Minute Match Outcome Prediction

### Description
Machine‑learning pipeline that predicts the winner of League of Legends
matches using the first 10 minutes of gameplay. The target is a binary
classification of whether the blue team wins.

### Key Components
- **Data**: ``data/high_diamond_ranked_10min.csv`` (downloaded from
  Kaggle).
- **Feature‑Engineering**: Removes identifier fields and perfectly
  collinear mirrored columns.
- **Exploratory Analysis**: Generates correlation plots, class‑balance
  charts, and feature‑importance visualisations.
- **Models**: Logistic Regression, Random Forest, XGBoost.
- **Evaluation**: Accuracy, Precision, Recall, F1‑Score, ROC‑AUC, ROC
  curves, confusion matrices.
- **Paper**: Academic write‑up in Springer LNNS style
  stored under ``paper/``.

### Artifacts
- ``main.py`` orchestrates data loading, EDA, modelling and
  result reporting.
- ``src/`` contains modular modules for data loading, EDA, and model
  training.
- All evaluated plots are placed in ``plots/``.
- Test set performance metrics are written to
  ``plots/model_metrics.csv``.

---

### Future Enhancements
- Integrate with an online prediction service.
- Replace logistic regression with a calibrated probability model.
- Explore real‑time streaming inference.
