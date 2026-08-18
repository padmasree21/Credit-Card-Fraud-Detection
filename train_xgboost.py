import os

import joblib
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.config import MODELS_DIR, RANDOM_SEED
from src.data_preprocessing import prepare_data
from src.evaluate import compute_metrics, print_metrics


def train_xgboost(X_train, y_train, tune: bool = True):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    if not tune:
        base_model.fit(X_train, y_train)
        return base_model

    param_grid = {
        "max_depth": [3, 5],
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
    }
    search = GridSearchCV(
        base_model, param_grid, scoring="average_precision", cv=3, n_jobs=-1
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    return search.best_estimator_


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    model = train_xgboost(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_score)
    print_metrics("XGBoost", metrics)

    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost.pkl"))
    return model, metrics


if __name__ == "__main__":
    main()
