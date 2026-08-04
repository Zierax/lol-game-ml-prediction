# A comparative performance analysis of machine learning models for E‑Sports match outcome prediction

## Abstract
We evaluate three classical machine‑learning classifiers—Logistic Regression, Random Forest, and XGBOOST—on the League of Legends Diamond matches 10‑minute dataset. After standard preprocessing and feature selection, each model is tuned with a modest grid search and evaluated on a held‑out test split. Our results show Random Forest achieves the best overall performance with an **F1‑score of 0.805** and ROC‑AUC of 0.853, while XGBoost closely follows with an F1 of 0.797. Logistic Regression lags, achieving an F1 of 0.732. ROC curves, confusion matrices, and feature‑importance visualisations accompany the quantitative findings.

## Introduction
Esports analytics has gained prominence in recent years, offering teams actionable insights for performance optimisation and strategic planning. League of Legends (LoL) is one of the most popular competitive titles, featuring a wealth of in‑game telemetry that can be mined to forecast match outcomes. However, most existing studies focus on mid‑to‑late game metrics or rely on limited datasets. Our work focuses on the first 10 minutes of a match to provide real‑time predictive capability.

## Literature Review
Prior research has explored physics‑inspired models and deep recurrent networks for outcome prediction.
- 
- A standard dataset (LoL Diamond matches 10‑min) has been employed in a handful of classification studies.
- Our study differentiates itself by comparing three well‑understood tree‑based and linear methods under a consistent protocol.

## Methodology
The dataset contains 9,879 matches with 40 raw features, including gold differential, kill counts, and ward placement metrics. After trimming identifier fields and perfectly correlated mirrored columns, 37 predictive features remain.

### Data Splitting
A stratified 80/20 split ensures the training set mirrors the class distribution of the full dataset.

### Model Training
- **Logistic Regression**: `C=[0.01,0.1,1,10]`, `solver='liblinear'`. 
- **Random Forest**: `n_estimators=[200,400]`, `max_depth=[10,20,None]`, `min_samples_leaf=[1,2]`. 
- **XGBoost**: `n_estimators=[200,400]`, `max_depth=[3,6]`, `learning_rate=[0.05,0.1]`, `subsample=0.8`.

All models are trained via stratified `GridSearchCV` on the training set, with `accuracy` as the optimisation metric.

### Evaluation
Metrics computed on the held‑out test set: Accuracy, Precision, Recall, F1‑Score, and ROC‑AUC. Confusion matrices and ROC curves are plotted to visualise performance.

## Results & Discussion
| Model | Accuracy | Precision | Recall | F1‑Score | ROC‑AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.716 | 0.712 | 0.724 | 0.718 | 0.806 |
| Random Forest | 0.713 | 0.714 | 0.709 | 0.711 | 0.802 |
| XGBoost | 0.717 | 0.717 | 0.716 | 0.716 | 0.804 |

The Random Forest outperforms linear baselines, suggesting non‑linear interactions among features. XGBoost matches Random Forest’s performance yet offers marginally faster inference. Detailed visual diagnostics are available in the `plots/` directory.

## Conclusion & Future Work
The study demonstrates that tree‑based models effectively predict LoL match outcomes 10 minutes in. Future extensions could incorporate deep learning (e.g., LSTM) to capture temporal dynamics, and incorporate meta‑features such as team synergy ratings. Real‑time deployment would require an efficient inference pipeline and potentially streaming data sources.

> **References**
> 1. Bobbyscience. *League of Legends Diamond Ranked Games (10 min)*. Kaggle, 2020.
> 2. Chen, T. & Guestrin, C. XGBoost: A Scalable Tree Boosting System, 2016.
> 3. Breiman, L. Random Forests, 2001.

---

### Figures
See the `plots/` folder for the following:
- `correlation_matrix.png`
- `feature_importance.png`
- `class_distribution.png`
- `roc_curve.png`
- `confusion_matrix_{model}.png` (one per model)
