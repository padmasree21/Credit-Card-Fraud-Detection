import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "creditcard.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

RANDOM_SEED = 42
TEST_SIZE = 0.2
TARGET_COL = "Class"

os.makedirs(MODELS_DIR, exist_ok=True)
