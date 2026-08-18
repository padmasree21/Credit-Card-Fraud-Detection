# Credit Card Fraud Detection

A machine learning system for detecting fraudulent credit card transactions using ensemble models with a focus on high-precision classification.

## Overview

- Trains and compares three models — **Logistic Regression**, **XGBoost**, and a **PyTorch MLP** — on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 frauds, highly imbalanced ~0.17%).
- Includes exploratory data analysis, feature scaling, class-imbalance handling, and hyperparameter tuning.
- Evaluation focuses on precision, recall, F1, and ROC-AUC rather than raw accuracy, since accuracy is meaningless on this imbalanced dataset.

## Dataset

The dataset is not included in this repo (108MB, licensed for research use). Download `creditcard.csv` from Kaggle and place it at:

```
data/creditcard.csv
```

Columns: `Time`, `V1`...`V28` (PCA-anonymized features), `Amount`, `Class` (1 = fraud, 0 = legitimate).

## Project structure
​```
credit_card_repo/
  data/              # place creditcard.csv here (not tracked in git)
  models/            # trained model artifacts get saved here
  notebooks/         # optional exploratory notebooks
  src/
    config.py              # paths, constants, random seed
    data_preprocessing.py  # load, scale, split, handle imbalance
    eda.py                 # exploratory data analysis + plots
    models.py              # PyTorch MLP architecture
    train_logistic_regression.py
    train_xgboost.py
    train_mlp.py
    evaluate.py            # shared metrics/plotting utilities
  main.py            # end-to-end pipeline: preprocess -> train all -> compare
  requirements.txt
  README.md
​```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run exploratory data analysis:

```bash
python -m src.eda
```

Run the full pipeline (preprocess, train all three models, compare results):

```bash
python main.py
```

Or run a single model:

```bash
python -m src.train_logistic_regression
python -m src.train_xgboost
python -m src.train_mlp
```

Trained models are saved to `models/`. A comparison table (precision, recall, F1, ROC-AUC) is printed at the end of `main.py`.

## Approach

1. **EDA** — inspect class imbalance, feature distributions, and correlations.
2. **Preprocessing** — scale `Time` and `Amount` (the only non-PCA features), stratified train/test split to preserve fraud ratio.
3. **Imbalance handling** — class-weighted loss/`class_weight="balanced"` for Logistic Regression and XGBoost's `scale_pos_weight`, and a weighted loss for the PyTorch MLP.
4. **Models**
   - Logistic Regression — fast, interpretable baseline.
   - XGBoost — gradient-boosted trees, tuned via grid search over `max_depth`, `n_estimators`, `learning_rate`.
   - PyTorch MLP — feed-forward network with dropout and batch norm.
5. **Evaluation** — precision, recall, F1, ROC-AUC, confusion matrix, and precision-recall curves (accuracy is not used as the primary metric given the ~0.17% fraud rate).

## Results

## Results

| Model               | Precision | Recall | F1     | ROC-AUC |
|----------------------|-----------|--------|--------|---------|
| Logistic Regression  | 0.0609    | 0.9184 | 0.1141 | 0.9722  |
| XGBoost               | 0.1029    | 0.9082 | 0.1848 | 0.9797  |
| PyTorch MLP           | 0.7788    | 0.8265 | 0.8020 | 0.9618  |

## License

MIT


## License

MIT
