import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DATA_PATH, TARGET_COL
from src.data_preprocessing import load_data


def run_eda(path: str = DATA_PATH):
    df = load_data(path)

    print("Shape:", df.shape)
    print("\nClass distribution:")
    print(df[TARGET_COL].value_counts())
    print("\nFraud rate: {:.4%}".format(df[TARGET_COL].mean()))
    print("\nMissing values:", df.isnull().sum().sum())
    print("\nAmount stats by class:")
    print(df.groupby(TARGET_COL)["Amount"].describe())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.countplot(x=TARGET_COL, data=df, ax=axes[0])
    axes[0].set_title("Class distribution (0 = legit, 1 = fraud)")
    axes[0].set_yscale("log")

    sns.boxplot(x=TARGET_COL, y="Amount", data=df, ax=axes[1])
    axes[1].set_title("Transaction amount by class")
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig("eda_overview.png", dpi=150)
    print("\nSaved plot to eda_overview.png")


if __name__ == "__main__":
    run_eda()
