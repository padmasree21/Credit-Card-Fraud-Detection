import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PATH, RANDOM_SEED, TARGET_COL, TEST_SIZE


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not path or not path.endswith(".csv"):
        raise ValueError("DATA_PATH must point to creditcard.csv")
    return pd.read_csv(path)


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """PCA features V1-V28 are already scaled; only Time and Amount need it."""
    df = df.copy()
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )


def prepare_data(path: str = DATA_PATH):
    df = load_data(path)
    df = scale_features(df)
    return split_data(df)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Train: {X_train.shape}, fraud rate: {y_train.mean():.4%}")
    print(f"Test:  {X_test.shape}, fraud rate: {y_test.mean():.4%}")
