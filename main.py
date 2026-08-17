import time
import csv
import os

from src.train_logistic_regression import main as run_logistic_regression
from src.train_xgboost import main as run_xgboost
from src.train_mlp import main as run_mlp

RESULTS_CSV = "results.csv"


def run_model(name, fn, results):
    print(f"\n>>> Training {name} ...")
    start = time.time()
    try:
        _, metrics = fn()
        results[name] = metrics
        elapsed = time.time() - start
        print(f">>> {name} done in {elapsed:.1f}s")
    except Exception as e:
        print(f">>> {name} FAILED: {e}")
        results[name] = None


def print_comparison(results):
    print("\n" + "=" * 60)
    print("Model comparison")
    print("=" * 60)
    header = f"{'Model':<22}{'Precision':>10}{'Recall':>10}{'F1':>10}{'ROC-AUC':>10}"
    print(header)
    for name, metrics in results.items():
        if metrics is None:
            print(f"{name:<22}{'FAILED':>40}")
            continue
        print(
            f"{name:<22}"
            f"{metrics['precision']:>10.4f}"
            f"{metrics['recall']:>10.4f}"
            f"{metrics['f1']:>10.4f}"
            f"{metrics['roc_auc']:>10.4f}"
        )


def save_results_csv(results, path=RESULTS_CSV):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Precision", "Recall", "F1", "ROC-AUC"])
        for name, metrics in results.items():
            if metrics is None:
                writer.writerow([name, "FAILED", "", "", ""])
            else:
                writer.writerow([
                    name,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['roc_auc']:.4f}",
                ])
    print(f"\nSaved results to {os.path.abspath(path)}")


def main():
    results = {}
    run_model("Logistic Regression", run_logistic_regression, results)
    run_model("XGBoost", run_xgboost, results)
    run_model("PyTorch MLP", run_mlp, results)

    print_comparison(results)
    save_results_csv(results)


if __name__ == "__main__":
    main()
