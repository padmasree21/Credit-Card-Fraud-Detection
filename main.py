from src.train_logistic_regression import main as run_logistic_regression
from src.train_mlp import main as run_mlp
from src.train_xgboost import main as run_xgboost


def main():
    results = {}

    _, results["Logistic Regression"] = run_logistic_regression()
    _, results["XGBoost"] = run_xgboost()
    _, results["PyTorch MLP"] = run_mlp()

    print("\n" + "=" * 60)
    print("Model comparison")
    print("=" * 60)
    header = f"{'Model':<22}{'Precision':>10}{'Recall':>10}{'F1':>10}{'ROC-AUC':>10}"
    print(header)
    for name, metrics in results.items():
        print(
            f"{name:<22}"
            f"{metrics['precision']:>10.4f}"
            f"{metrics['recall']:>10.4f}"
            f"{metrics['f1']:>10.4f}"
            f"{metrics['roc_auc']:>10.4f}"
        )


if __name__ == "__main__":
    main()
