# LoL 10-Minute Match Outcome Prediction

Machine-learning pipeline that predicts the winner of League of Legends matches using the first 10 minutes of gameplay.

## Conference Submission

This work has been submitted to **ICDPN 2026** (International Conference on Data Processing and Networking) in Springer LNCS format.

- **Paper**: `paper/main.tex` (LaTeX) / `paper/main.pdf`
- **Cover Letter**: `paper/cover_letter.tex` / `paper/cover_letter.pdf`
- **Status**: Submitted (double-blind review)

## Project Overview

The target is a binary classification of whether the blue team wins based on early-game features at the 10-minute mark.

## Key Components

| Component | Description |
|-----------|-------------|
| **Data** | `data/high_diamond_ranked_10min.csv` (Kaggle) |
| **Preprocessing** | Removes identifiers and collinear mirrored columns |
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC |

## Results

All three models achieve comparable F1-scores (0.711-0.718) and ROC-AUC values (0.802-0.806), suggesting that early-game features contain limited discriminative information.

## Files

| File | Purpose |
|------|--------|
| `main.py` | Main pipeline orchestration |
| `src/data_loader.py` | Data loading and preprocessing |
| `src/eda.py` | Exploratory data analysis |
| `src/models.py` | Model training and evaluation |
| `paper/main.tex` | LaTeX paper (Springer LNCS) |
| `paper/cover_letter.tex` | Conference cover letter |
| `plots/` | Generated visualizations and metrics |

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Future Work

- Incorporate later time windows to assess when non-linear models outperform linear baselines
- Apply deep learning (LSTMs) for temporal dynamics
- Integrate real-time streaming inference
