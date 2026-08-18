import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODELS_DIR, RANDOM_SEED
from src.data_preprocessing import prepare_data
from src.evaluate import compute_metrics, print_metrics
from src.models import FraudMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor_loader(X, y, batch_size, shuffle):
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_mlp(X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=256, lr=1e-3):
    torch.manual_seed(RANDOM_SEED)

    train_loader = _to_tensor_loader(X_train, y_train, batch_size, shuffle=True)

    model = FraudMLP(input_dim=X_train.shape[1]).to(DEVICE)
    pos_weight = torch.tensor((y_train == 0).sum() / (y_train == 1).sum(), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"epoch {epoch:2d}/{epochs} - loss: {avg_loss:.4f}")

    return model


def predict_mlp(model, X, batch_size=512):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X.values, dtype=torch.float32)),
        batch_size=batch_size,
    )
    scores = []
    with torch.no_grad():
        for (X_batch,) in loader:
            logits = model(X_batch.to(DEVICE))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    model = train_mlp(X_train, y_train)

    y_score = predict_mlp(model, X_test)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_score)
    print_metrics("PyTorch MLP", metrics)

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "mlp.pt"))
    return model, metrics


if __name__ == "__main__":
    main()
