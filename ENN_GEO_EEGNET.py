from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Data IO (same style as your baseline script)
# ============================================================
def load_npz_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    pack = np.load(str(path), allow_pickle=True)
    return pack["X"].astype(np.float32), pack["y"].astype(np.int64)


def keep_4class_only(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = (y >= 1) & (y <= 4)
    X = X[m]
    y = y[m].astype(np.int64) - 1  # {1,2,3,4}->{0..3}
    return X, y


def keep_nc_only(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 0]


def zscore_per_trial(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # X: (N,1,C,T)
    m = X.mean(axis=-1, keepdims=True)
    s = X.std(axis=-1, keepdims=True)
    return (X - m) / (s + eps)


# ============================================================
# EEGNet backbone that exposes features (aligned with your baseline blocks)
# ============================================================
class EEGNetBackbone(nn.Module):
    """
    This follows your baseline EEGNet blocks, but returns features before classifier.
    """
    def __init__(
        self,
        n_ch: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_len: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
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

        feat = torch.flatten(x, start_dim=1)
        return feat


class EEGNetSoftmaxENN(nn.Module):
    """
    One shared EEGNet backbone, two heads:
      - softmax logits head (for baseline)
      - ENN head: evidence -> Dirichlet alpha
    """
    def __init__(self, n_ch: int, n_classes: int = 4, **eegnet_kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = EEGNetBackbone(n_ch=n_ch, **eegnet_kwargs)

        self.logit_head = None  # lazy init
        self.enn_head = None    # lazy init

    def _lazy_init(self, feat_dim: int, device: torch.device):
        if self.logit_head is None:
            self.logit_head = nn.Linear(feat_dim, self.n_classes).to(device)
        if self.enn_head is None:
            self.enn_head = nn.Linear(feat_dim, self.n_classes).to(device)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        self._lazy_init(feat.shape[1], feat.device)
        return feat

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.logit_head(feat)

    def forward_alpha(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        evidence = F.softplus(self.enn_head(feat))     # >=0
        alpha = evidence + 1.0
        return alpha


# ============================================================
# EDL / ENN losses
# ============================================================
def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)
    beta = torch.ones_like(alpha)

    lgammaK = torch.lgamma(torch.tensor(float(K), device=alpha.device))
    kl = (
        torch.lgamma(S)
        - torch.sum(torch.lgamma(alpha), dim=1)
        - lgammaK
        + torch.sum((alpha - beta) * (digamma(alpha) - digamma(S)), dim=1)
    )
    return kl.mean()


def edl_id_loss(alpha: torch.Tensor, target: torch.Tensor, kl_weight: float = 1e-3) -> torch.Tensor:
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)
    y = F.one_hot(target, num_classes=K).float()
    ece = torch.sum(y * (digamma(S) - digamma(alpha)), dim=1)  # expected CE under Dirichlet
    kl = dirichlet_kl_to_uniform(alpha)
    return ece.mean() + kl_weight * kl


def ood_low_evidence_loss(alpha: torch.Tensor, mode: str = "kl") -> torch.Tensor:
    # Encourage low evidence on NC (OOD) samples
    if mode == "kl":
        return dirichlet_kl_to_uniform(alpha)
    if mode == "smean":
        return alpha.sum(dim=1).mean()
    raise ValueError(mode)


# ============================================================
# Geometry (Mahalanobis) + ENN+GEO fusion
# ============================================================
def fit_maha_shared(Ftr: np.ndarray, ytr: np.ndarray, K: int, shrink: float = 1e-2):
    D = Ftr.shape[1]
    mu = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        mu[k] = Ftr[ytr == k].mean(axis=0)

    Xc = Ftr - mu[ytr]
    cov = (Xc.T @ Xc) / max(len(Ftr) - 1, 1)
    cov = cov + shrink * np.eye(D, dtype=np.float32)
    inv_cov = np.linalg.inv(cov).astype(np.float32)
    return mu.astype(np.float32), inv_cov


def maha_d2_all(Fx: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    N, D = Fx.shape
    K = mu.shape[0]
    d2 = np.empty((N, K), dtype=np.float32)
    for k in range(K):
        diff = Fx - mu[k][None, :]
        d2[:, k] = np.sum((diff @ inv_cov) * diff, axis=1)
    return d2


def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=axis, keepdims=True) + 1e-12)


def geo_weights_from_d2_softmax(d2: np.ndarray, tau: float = 10.0) -> np.ndarray:
    # smaller d2 => larger weight
    return softmax_np(-d2 / tau, axis=1)


@torch.no_grad()
def collect_scores_ic(
    model: EEGNetSoftmaxENN,
    X_ic: np.ndarray,
    y_ic: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    tau_geo: float = 10.0,
    bs: int = 256,
) -> Dict[str, np.ndarray]:
    """
    returns:
      y_true
      pred_soft, pmax
      pred_enn, S
      S_geo
    """
    model.eval()
    y_true_list = []
    pred_soft_list, pmax_list = [], []
    pred_enn_list, S_list = [], []
    Sgeo_list = []

    for i in range(0, len(X_ic), bs):
        xb = torch.from_numpy(X_ic[i:i+bs]).float().to(DEVICE)
        yb = y_ic[i:i+bs]

        # softmax head
        logits = model.forward_logits(xb)
        prob = torch.softmax(logits, dim=1)
        p_max, pred_s = torch.max(prob, dim=1)

        # ENN head
        alpha = model.forward_alpha(xb)
        S = alpha.sum(dim=1)
        prob_enn = alpha / (S[:, None] + 1e-12)
        pred_e = torch.argmax(prob_enn, dim=1)

        # GEO-fused strength
        feat = model.forward_features(xb).cpu().numpy()
        d2_all = maha_d2_all(feat, mu, inv_cov)                 # (B,K)
        w = geo_weights_from_d2_softmax(d2_all, tau=tau_geo)    # (B,K)

        evidence = (alpha - 1.0).cpu().numpy().astype(np.float32)  # (B,K)
        # fused strength: S_geo = sum_k (w_k * evidence_k + 1)
        S_geo = np.sum(evidence * w + 1.0, axis=1).astype(np.float32)

        y_true_list.append(yb)
        pred_soft_list.append(pred_s.cpu().numpy())
        pmax_list.append(p_max.cpu().numpy())

        pred_enn_list.append(pred_e.cpu().numpy())
        S_list.append(S.cpu().numpy())

        Sgeo_list.append(S_geo)

    return dict(
        y_true=np.concatenate(y_true_list),
        pred_soft=np.concatenate(pred_soft_list),
        pmax=np.concatenate(pmax_list).astype(np.float32),
        pred_enn=np.concatenate(pred_enn_list),
        S=np.concatenate(S_list).astype(np.float32),
        S_geo=np.concatenate(Sgeo_list).astype(np.float32),
    )


@torch.no_grad()
def collect_scores_ood(
    model: EEGNetSoftmaxENN,
    X_ic: np.ndarray,
    X_nc: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    tau_geo: float = 10.0,
    bs: int = 256,
) -> Dict[str, float]:
    """
    OOD detection IC vs NC (y_true: 0 for IC, 1 for NC)
      - softmax: score = 1 - pmax (higher => more OOD)
      - ENN: score = -S (higher => more OOD)
      - ENN+GEO: score = -S_geo (higher => more OOD)
      - GEO only: score = dmin (higher => more OOD)
    """
    model.eval()

    def batch_scores(X: np.ndarray):
        pmaxs, Ss, Sgeos, dmins = [], [], [], []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).float().to(DEVICE)

            logits = model.forward_logits(xb)
            prob = torch.softmax(logits, dim=1)
            p_max = torch.max(prob, dim=1).values

            alpha = model.forward_alpha(xb)
            S = alpha.sum(dim=1)

            feat = model.forward_features(xb).cpu().numpy()
            d2_all = maha_d2_all(feat, mu, inv_cov)
            dmin = d2_all.min(axis=1)

            w = geo_weights_from_d2_softmax(d2_all, tau=tau_geo)
            evidence = (alpha - 1.0).cpu().numpy().astype(np.float32)
            S_geo = np.sum(evidence * w + 1.0, axis=1).astype(np.float32)

            pmaxs.append(p_max.cpu().numpy())
            Ss.append(S.cpu().numpy())
            Sgeos.append(S_geo)
            dmins.append(dmin.astype(np.float32))
        return (
            np.concatenate(pmaxs).astype(np.float32),
            np.concatenate(Ss).astype(np.float32),
            np.concatenate(Sgeos).astype(np.float32),
            np.concatenate(dmins).astype(np.float32),
        )

    pmax_ic, S_ic, Sgeo_ic, dmin_ic = batch_scores(X_ic)
    pmax_nc, S_nc, Sgeo_nc, dmin_nc = batch_scores(X_nc)

    y_true = np.concatenate([np.zeros(len(X_ic), dtype=np.int64), np.ones(len(X_nc), dtype=np.int64)])

    score_soft = np.concatenate([1.0 - pmax_ic, 1.0 - pmax_nc])    # higher => OOD
    score_enn  = np.concatenate([-S_ic, -S_nc])                    # higher => OOD
    score_geoF = np.concatenate([-Sgeo_ic, -Sgeo_nc])              # higher => OOD
    score_dmin = np.concatenate([dmin_ic, dmin_nc])                # higher => OOD

    return dict(
        auroc_soft=float(roc_auc_score(y_true, score_soft)),
        auroc_enn=float(roc_auc_score(y_true, score_enn)),
        auroc_geoF=float(roc_auc_score(y_true, score_geoF)),
        auroc_dmin=float(roc_auc_score(y_true, score_dmin)),
    )


# ============================================================
# Rejection by COVERAGE (crucial for fair comparison)
# ============================================================
def rejection_by_coverage(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, coverages: List[float]):
    """
    conf: higher => more confident
    accept top-k by conf to match target coverage
    """
    N = len(conf)
    order = np.argsort(-conf)  # descending
    rows = []
    for c in coverages:
        k = int(round(float(c) * N))
        k = max(min(k, N), 0)
        accept = np.zeros(N, dtype=bool)
        if k > 0:
            accept[order[:k]] = True
            acc = float((y_pred[accept] == y_true[accept]).mean())
        else:
            acc = float("nan")
        rows.append((float(c), float(accept.mean()), int(k), acc))
    return rows


def print_rejection_table(name: str, rows):
    print(f"\n[IC Rejection] {name}")
    print("target_cov\taccept_rate\taccepted_n\tacc@accepted")
    for c, ar, n, acc in rows:
        acc_str = "nan" if np.isnan(acc) else f"{acc:.4f}"
        print(f"{c:.2f}\t\t{ar:.3f}\t\t{n}\t\t{acc_str}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=r"processed_final\win2.0s")
    ap.add_argument("--subject", type=str, default="A01", help="A01..A09")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=DEVICE)
    ap.add_argument("--no_zscore", action="store_true")

    # ENN/NC settings
    ap.add_argument("--lambda_enn", type=float, default=0.2, help="weight for EDL ID loss (in addition to CE)")
    ap.add_argument("--lambda_ood", type=float, default=0.15, help="weight for NC low-evidence loss")
    ap.add_argument("--nc_train_ratio", type=float, default=0.10, help="use 10% NC from T_train for OOD calibration")
    ap.add_argument("--nc_test_ratio", type=float, default=0.90, help="use 90% NC from E_test for OOD test")
    ap.add_argument("--kl_weight_id", type=float, default=1e-3)

    # GEO settings
    ap.add_argument("--tau_geo", type=float, default=10.0)
    ap.add_argument("--shrink", type=float, default=1e-2)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    base = Path(args.base_dir)
    sub = args.subject

    # STRICT protocol (same as your baseline script)
    X_tr_all, y_tr_all = load_npz_xy(base / f"{sub}T_train.npz")
    X_va_all, y_va_all = load_npz_xy(base / f"{sub}T_val.npz")
    X_te_all, y_te_all = load_npz_xy(base / f"{sub}E_test.npz")

    # IC
    X_tr, y_tr = keep_4class_only(X_tr_all, y_tr_all)
    X_va, y_va = keep_4class_only(X_va_all, y_va_all)
    X_te, y_te = keep_4class_only(X_te_all, y_te_all)

    # NC pools
    X_nc_tr = keep_nc_only(X_tr_all, y_tr_all)  # from T_train only
    X_nc_te = keep_nc_only(X_te_all, y_te_all)  # from E_test only

    # z-score
    if not args.no_zscore:
        X_tr = zscore_per_trial(X_tr)
        X_va = zscore_per_trial(X_va)
        X_te = zscore_per_trial(X_te)
        if len(X_nc_tr) > 0:
            X_nc_tr = zscore_per_trial(X_nc_tr)
        if len(X_nc_te) > 0:
            X_nc_te = zscore_per_trial(X_nc_te)

    # NC split: 10% calib from train NC, 90% test from E_test NC
    rng = np.random.default_rng(args.seed)
    if len(X_nc_tr) > 0:
        idx = rng.permutation(len(X_nc_tr))
        m = int(round(len(idx) * float(args.nc_train_ratio)))
        X_nc_cal = X_nc_tr[idx[:m]]
    else:
        X_nc_cal = np.zeros((0,) + X_tr.shape[1:], dtype=np.float32)

    if len(X_nc_te) > 0:
        idx = rng.permutation(len(X_nc_te))
        m = int(round(len(idx) * float(args.nc_test_ratio)))
        X_nc_test = X_nc_te[idx[:m]]
    else:
        X_nc_test = np.zeros((0,) + X_tr.shape[1:], dtype=np.float32)

    print(f"[{sub}] IC Train(T_train): X={X_tr.shape}, y={np.bincount(y_tr, minlength=4)}")
    print(f"[{sub}] IC   Val(T_val):   X={X_va.shape}, y={np.bincount(y_va, minlength=4)}")
    print(f"[{sub}] IC  Test(E_test):  X={X_te.shape}, y={np.bincount(y_te, minlength=4)}")
    print(f"[{sub}] NC calib (T_train {args.nc_train_ratio:.2f}): {len(X_nc_cal)} | NC test (E_test {args.nc_test_ratio:.2f}): {len(X_nc_test)}")
    
    # batch
    def next_nc_batch():
        nonlocal nc_iter
        if nc_loader is None:
            return None
        try:
            (xnc,) = next(nc_iter)
        except StopIteration:
            nc_iter = iter(nc_loader)
            (xnc,) = next(nc_iter)
        return xnc


    n_ch = X_tr.shape[2]
    model = EEGNetSoftmaxENN(n_ch=n_ch, n_classes=4).to(device)

    # ---- IMPORTANT: force init heads BEFORE creating optimizer ----
    with torch.no_grad():
        xb0 = torch.from_numpy(X_tr[:2]).float().to(device)  # any small batch
        _ = model.forward_logits(xb0)
        _ = model.forward_alpha(xb0)


    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        batch_size=args.batch, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long()),
        batch_size=args.batch, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long()),
        batch_size=args.batch, shuffle=False
    )

    if len(X_nc_cal) >= 1:
        nc_bs = min(args.batch, len(X_nc_cal))   # 关键：别超过样本数
        nc_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_nc_cal).float()),
            batch_size=nc_bs,
            shuffle=True,
            drop_last=False,                      # 关键：不要丢掉最后一个小 batch
        )
        nc_iter = iter(nc_loader)
    else:
        nc_loader, nc_iter = None, None


    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def eval_softmax_acc(loader):
        model.eval()
        corr, tot = 0, 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model.forward_logits(xb)
            pred = torch.argmax(logits, dim=1)
            corr += int((pred == yb).sum().item())
            tot += int(yb.numel())
        return corr / max(tot, 1)

    @torch.no_grad()
    def eval_enn_acc(loader):
        model.eval()
        corr, tot = 0, 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            alpha = model.forward_alpha(xb)
            S = alpha.sum(dim=1, keepdim=True)
            prob = alpha / (S + 1e-12)
            pred = torch.argmax(prob, dim=1)
            corr += int((pred == yb).sum().item())
            tot += int(yb.numel())
        return corr / max(tot, 1)

    best_val = -1.0
    best_state = None
    best_ep = -1
    bad = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)

            logits = model.forward_logits(xb)
            alpha = model.forward_alpha(xb)

            loss_ce = ce(logits, yb)
            loss_edl = edl_id_loss(alpha, yb, kl_weight=float(args.kl_weight_id))
            loss = loss_ce + float(args.lambda_enn) * loss_edl

            # NC calibration: low evidence on NC (OOD)

            xnc = next_nc_batch()
            if xnc is not None and float(args.lambda_ood) > 0:
                xnc = xnc.to(device)
                alpha_nc = model.forward_alpha(xnc)
                loss_ood = ood_low_evidence_loss(alpha_nc, mode="kl")
                loss = loss + float(args.lambda_ood) * loss_ood


            loss.backward()
            opt.step()

        # early stop by VAL accuracy (softmax val, consistent with your baseline habit)
        val_acc = eval_softmax_acc(val_loader)
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
            bad = 0
        else:
            bad += 1

        if ep == 1 or ep % 10 == 0:
            te_soft = eval_softmax_acc(test_loader)
            te_enn = eval_enn_acc(test_loader)
            print(f"epoch {ep:03d} | val_soft={val_acc:.4f} | test_soft={te_soft:.4f} | test_enn={te_enn:.4f} | best_val={best_val:.4f}@{best_ep}")
        if bad >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # ============================================================
    # Fit GEO on TRAIN(IC) features
    # ============================================================
    model.eval()
    Ftr = []
    for i in range(0, len(X_tr), 256):
        xb = torch.from_numpy(X_tr[i:i+256]).float().to(device)
        feat = model.forward_features(xb).detach().cpu().numpy()
        Ftr.append(feat)
    Ftr = np.concatenate(Ftr, axis=0)
    mu, inv_cov = fit_maha_shared(Ftr, y_tr, K=4, shrink=float(args.shrink))

    # ============================================================
    # IC rejection comparison (coverage-aligned)
    # ============================================================
    scores_ic = collect_scores_ic(model, X_te, y_te, mu, inv_cov, tau_geo=float(args.tau_geo))

    # forced acc reference
    forced_soft = float((scores_ic["pred_soft"] == scores_ic["y_true"]).mean())
    forced_enn  = float((scores_ic["pred_enn"] == scores_ic["y_true"]).mean())
    print(f"\n[{sub}] FORCED IC acc: Softmax={forced_soft:.4f} | ENN(prob)={forced_enn:.4f}")

    coverages = [1.0,0.9,0.8,0.7,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1]
    rows_soft = rejection_by_coverage(scores_ic["y_true"], scores_ic["pred_soft"], scores_ic["pmax"], coverages)
    rows_enn  = rejection_by_coverage(scores_ic["y_true"], scores_ic["pred_enn"],  scores_ic["S"], coverages)
    rows_geo  = rejection_by_coverage(scores_ic["y_true"], scores_ic["pred_enn"],  scores_ic["S_geo"], coverages)

    print_rejection_table("EEGNet Softmax (conf=p_max)", rows_soft)
    print_rejection_table("EEGNet-ENN (conf=S)", rows_enn)
    print_rejection_table("EEGNet-ENN+GEO (conf=S_geo)", rows_geo)

    # ============================================================
    # OOD eval: IC vs NC_test (optional but useful)
    # ============================================================
    if len(X_nc_test) > 0:
        ood = collect_scores_ood(model, X_te, X_nc_test, mu, inv_cov, tau_geo=float(args.tau_geo))
        print("\n[OOD AUROC] IC (E_test IC) vs NC (E_test NC_test)")
        print(f"  Softmax (1-pmax):     {ood['auroc_soft']:.4f}")
        print(f"  ENN (-S):             {ood['auroc_enn']:.4f}")
        print(f"  ENN+GEO (-S_geo):     {ood['auroc_geoF']:.4f}")
        print(f"  GEO only (dmin):      {ood['auroc_dmin']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
