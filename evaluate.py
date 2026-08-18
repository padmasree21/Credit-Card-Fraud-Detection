import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for key, value in metrics.items():
        print(f"{key:>10}: {value:.4f}")


def plot_confusion_matrix(y_true, y_pred, name: str, save_path: str | None = None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion matrix - {name}")
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_precision_recall(y_true, y_score, name: str, save_path: str | None = None):
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_score, name=name)
    plt.title(f"Precision-Recall curve - {name}")
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
