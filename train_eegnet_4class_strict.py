from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -------------------------
# Loading + label filtering
# -------------------------
def load_npz_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    pack = np.load(str(path), allow_pickle=True)
    return pack["X"], pack["y"]


def keep_4class_only(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = (y >= 1) & (y <= 4)
    X = X[mask]
    y = y[mask].astype(np.int64) - 1  # {1,2,3,4}->{0,1,2,3}
    return X, y


def zscore_per_trial(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # X: (N,1,C,T)
    m = X.mean(axis=-1, keepdims=True)
    s = X.std(axis=-1, keepdims=True)
    return (X - m) / (s + eps)


# -------------------------
# EEGNet (classic baseline)
# -------------------------
class EEGNet(nn.Module):
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
        self.n_classes = n_classes

        self.conv_temporal = nn.Conv2d(1, F1, kernel_size=(1, kernel_len), padding=(0, kernel_len // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv_depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_dw = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sep_pw = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        self.classifier = None  # init lazily

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_temporal(x)
        x = self.bn1(x)

        x = self.conv_depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_dw(x)
        x = self.sep_pw(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = torch.flatten(x, start_dim=1)
        if self.classifier is None:
            self.classifier = nn.Linear(x.shape[1], self.n_classes).to(x.device)
        return self.classifier(x)


# -------------------------
# Train / eval
# -------------------------
@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


@torch.no_grad()
def collect_softmax_maxprob(model: nn.Module, loader: DataLoader, device: str):
    """
    return:
      y_true: (N,)
      y_pred: (N,)
      p_max : (N,)   max softmax prob
    """
    model.eval()
    ys, preds, pmaxs = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)
        p_max, pred = torch.max(prob, dim=1)

        ys.append(yb.cpu().numpy())
        preds.append(pred.cpu().numpy())
        pmaxs.append(p_max.cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    p_max = np.concatenate(pmaxs, axis=0)
    return y_true, y_pred, p_max


def rejection_curve_from_pmax(y_true: np.ndarray, y_pred: np.ndarray, p_max: np.ndarray, taus):
    """
    For each tau:
      accept = p_max >= tau
      acceptance_rate = mean(accept)
      acc_on_accepted = accuracy restricted to accepted (None if 0 accepted)
    """
    rows = []
    for tau in taus:
        accept = p_max >= float(tau)
        ar = float(accept.mean())
        n_acc = int(accept.sum())
        if n_acc == 0:
            acc = float("nan")
        else:
            acc = float((y_pred[accept] == y_true[accept]).mean())
        rows.append((float(tau), ar, n_acc, acc))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=r"processed_final\win2.0s")
    parser.add_argument("--subject", type=str, default="A01", help="A01..A09")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_zscore", action="store_true")
    parser.add_argument("--rejection", action="store_true", help="run rejection evaluation on E_test using softmax max-prob")
    parser.add_argument("--taus", type=str, default="0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9",
                        help="comma-separated thresholds for rejection on p_max")



    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base = Path(args.base_dir)
    sub = args.subject

    # STRICT protocol:
    # Train: T_train
    # Val:   T_val (early stopping)
    # Test:  E_test (final)
    X_tr, y_tr = load_npz_xy(base / f"{sub}T_train.npz")
    X_va, y_va = load_npz_xy(base / f"{sub}T_val.npz")
    X_te, y_te = load_npz_xy(base / f"{sub}E_test.npz")

    X_tr, y_tr = keep_4class_only(X_tr, y_tr)
    X_va, y_va = keep_4class_only(X_va, y_va)
    X_te, y_te = keep_4class_only(X_te, y_te)

    if not args.no_zscore:
        X_tr = zscore_per_trial(X_tr)
        X_va = zscore_per_trial(X_va)
        X_te = zscore_per_trial(X_te)

    print(f"[{sub}] Train(T_train): X={X_tr.shape}, y={np.bincount(y_tr, minlength=4)}")
    print(f"[{sub}]   Val(T_val):   X={X_va.shape}, y={np.bincount(y_va, minlength=4)}")
    print(f"[{sub}]  Test(E_test):  X={X_te.shape}, y={np.bincount(y_te, minlength=4)}")

    device = args.device
    n_ch = X_tr.shape[2]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
                              batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long()),
                            batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long()),
                             batch_size=args.batch, shuffle=False)

    model = EEGNet(n_ch=n_ch, n_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val = -1.0
    best_test_at_best_val = -1.0
    best_epoch = -1
    bad = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        val_acc = eval_acc(model, val_loader, device=device)
        test_acc = eval_acc(model, test_loader, device=device)

        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_test_at_best_val = test_acc
            best_epoch = ep
            bad = 0
        else:
            bad += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:03d} | val_acc={val_acc:.4f} | test_acc={test_acc:.4f} | best_val={best_val:.4f}@{best_epoch} -> test={best_test_at_best_val:.4f}")

        if bad >= args.patience:
            break

    print(f"[{sub}] BEST (selected by T_val): val_acc={best_val:.4f} @ epoch {best_epoch} | FINAL REPORT test_acc(E_test)={best_test_at_best_val:.4f}")
    
    # -------------------------
# Rejection evaluation (selective classification)
# -------------------------
    if args.rejection:
        taus = [float(x.strip()) for x in args.taus.split(",") if x.strip()]
        y_true, y_pred, p_max = collect_softmax_maxprob(model, test_loader, device=device)
        rows = rejection_curve_from_pmax(y_true, y_pred, p_max, taus)

        print("\n[Rejection eval on E_test] score = max softmax prob (p_max)")
        print("tau\taccept_rate\taccepted_n\tacc@accepted")
        for tau, ar, n_acc, acc in rows:
            acc_str = "nan" if np.isnan(acc) else f"{acc:.4f}"
            print(f"{tau:.2f}\t{ar:.3f}\t\t{n_acc}\t\t{acc_str}")

        # (optional) also print best acc among taus (ignoring nan)
        valid = [(tau, ar, n_acc, acc) for tau, ar, n_acc, acc in rows if not np.isnan(acc)]
        if valid:
            best = max(valid, key=lambda r: r[3])
            print(f"\n[Best acc@accepted] tau={best[0]:.2f} | accept_rate={best[1]:.3f} | accepted_n={best[2]} | acc@accepted={best[3]:.4f}")


if __name__ == "__main__":
    main()
