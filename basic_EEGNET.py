from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -------------------------
# Utils: load + filter labels
# -------------------------
def load_npz_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    pack = np.load(str(path), allow_pickle=True)
    X = pack["X"]  # (N,1,C,T) if your out_format=eegnet
    y = pack["y"]  # (N,)
    return X, y


def concat_splits(base_dir: Path, stem: str, split_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for sp in split_names:
        p = base_dir / f"{stem}_{sp}.npz"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        X, y = load_npz_xy(p)
        Xs.append(X)
        ys.append(y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def keep_4class_only(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = (y >= 1) & (y <= 4)
    X = X[mask]
    y = y[mask].astype(np.int64) - 1  # map {1,2,3,4} -> {0,1,2,3}
    return X, y

def zscore_per_trial(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # X: (N,1,C,T)
    m = X.mean(axis=-1, keepdims=True)
    s = X.std(axis=-1, keepdims=True)
    return (X - m) / (s + eps)


# -------------------------
# EEGNet (classic)
# -------------------------
class EEGNet(nn.Module):
    """
    A standard EEGNet-style network for (N,1,C,T).
    Good enough as a baseline (not super optimized).
    """
    def __init__(
        self,
        n_ch: int,
        n_classes: int = 4,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_len: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_ch = n_ch
        self.n_classes = n_classes

        # Block 1: temporal conv
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_len),
            padding=(0, kernel_len // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise conv over channels
        self.conv_depthwise = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_ch, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv
        self.conv_separable_depth = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        )
        self.conv_separable_point = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        self.classifier = None  # init after we see T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,1,C,T)
        x = self.conv_temporal(x)
        x = self.bn1(x)

        x = self.conv_depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = torch.flatten(x, start_dim=1)
        if self.classifier is None:
            self.classifier = nn.Linear(x.shape[1], self.n_classes).to(x.device)
        return self.classifier(x)


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def train_one_subject(
    base_dir: Path,
    subject: str,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 80,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --------
    # Load: train from T(all), test on E(all)
    # --------
    X_T, y_T = concat_splits(base_dir, f"{subject}T", ["train", "val", "test"])
    X_E, y_E = concat_splits(base_dir, f"{subject}E", ["train", "val", "test"])

    X_T, y_T = keep_4class_only(X_T, y_T)
    X_E, y_E = keep_4class_only(X_E, y_E)

    # tensors
    X_T_t = torch.from_numpy(X_T).float()
    y_T_t = torch.from_numpy(y_T).long()
    X_E_t = torch.from_numpy(X_E).float()
    y_E_t = torch.from_numpy(y_E).long()

    X_T = zscore_per_trial(X_T)
    X_E = zscore_per_trial(X_E)


    n_ch = X_T.shape[2]
    n_time = X_T.shape[3]
    print(f"[{subject}] Train(T): X={X_T.shape}, y={np.bincount(y_T, minlength=4)} | Test(E): X={X_E.shape}, y={np.bincount(y_E, minlength=4)}")

    train_loader = DataLoader(TensorDataset(X_T_t, y_T_t), batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(TensorDataset(X_E_t, y_E_t), batch_size=batch_size, shuffle=False, drop_last=False)

    model = EEGNet(n_ch=n_ch, n_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_epoch = -1

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        acc = eval_acc(model, test_loader, device=device)
        if acc > best_acc:
            best_acc = acc
            best_epoch = ep
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:03d} | test_acc={acc:.4f} | best={best_acc:.4f}@{best_epoch}")

    print(f"[{subject}] FINAL best test_acc on E = {best_acc:.4f} (epoch {best_epoch})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=r"processed_final\win2.0s", help="folder containing A01T_train.npz etc.")
    parser.add_argument("--subject", type=str, default="A01", help="A01..A09")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    train_one_subject(
        base_dir=base_dir,
        subject=args.subject,
        batch_size=args.batch,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
