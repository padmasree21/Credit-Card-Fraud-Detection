import joblib
from sklearn.linear_model import LogisticRegression

from src.config import MODELS_DIR, RANDOM_SEED
from src.data_preprocessing import prepare_data
from src.evaluate import compute_metrics, print_metrics

import os


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    model = train_logistic_regression(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_score)
    print_metrics("Logistic Regression", metrics)

    joblib.dump(model, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    return model, metrics


if __name__ == "__main__":
    main()
