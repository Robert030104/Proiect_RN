from pathlib import Path
import pickle
import json
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

from model import MLP


def get_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_scaler(root: Path, scaler_rel: str = "config/scaler.pkl"):
    scaler_path = (root / scaler_rel).resolve()
    if not scaler_path.exists():
        raise FileNotFoundError(f"Lipseste: {scaler_path}")
    with open(scaler_path, "rb") as f:
        sc = pickle.load(f)
    feature_names = list(sc["feature_names"])
    mean = np.array(sc["mean"], dtype=np.float32)
    scale = np.array(sc["scale"], dtype=np.float32)
    return scaler_path, feature_names, mean, scale


def _find_xy_splits(root: Path):
    trX = root / "data" / "train" / "X_train.csv"
    trY = root / "data" / "train" / "y_train.csv"

    vaX = root / "data" / "validation" / "X_val.csv"
    vaY = root / "data" / "validation" / "y_val.csv"

    teX = root / "data" / "test" / "X_test.csv"
    teY = root / "data" / "test" / "y_test.csv"

    missing = []
    for p in [trX, trY, vaX, vaY, teX, teY]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Lipsesc fisierele split (X_/y_):\n- " + "\n- ".join(missing))

    return (trX, trY), (vaX, vaY), (teX, teY)


def _load_xy(x_path: Path, y_path: Path, feature_names, mean, scale):
    Xdf = pd.read_csv(x_path)
    ydf = pd.read_csv(y_path)

    if "defect" in ydf.columns:
        y = pd.to_numeric(ydf["defect"], errors="coerce").fillna(0).astype(int).values.astype(np.float32)
    else:
        y = pd.to_numeric(ydf.iloc[:, 0], errors="coerce").fillna(0).astype(int).values.astype(np.float32)

    y = np.where(y >= 0.5, 1.0, 0.0).astype(np.float32)

    missing = [c for c in feature_names if c not in Xdf.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane in {x_path}: {missing}")

    X = Xdf[feature_names].copy()
    for c in feature_names:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    X = X.values.astype(np.float32)
    X = (X - mean) / (scale + 1e-12)
    return X, y


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.55, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        targets = targets.view(-1)
        logits = logits.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = targets * p + (1.0 - targets) * (1.0 - p)
        w = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = w * ((1.0 - pt) ** self.gamma) * bce
        return loss.mean()


def eval_auc_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    probs_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb).view(-1)
            loss = criterion(logits, yb.view(-1))
            losses.append(float(loss.item()))
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            probs_all.append(prob)
            y_all.append(yb.detach().cpu().numpy())

    p = np.concatenate(probs_all, axis=0).reshape(-1)
    y = np.concatenate(y_all, axis=0).reshape(-1)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    val_loss = float(np.mean(losses)) if losses else 0.0
    return auc, val_loss


def get_probs(model, loader, device):
    model.eval()
    probs_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb).view(-1)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            probs_all.append(prob)
            y_all.append(yb.detach().cpu().numpy())
    p = np.concatenate(probs_all, axis=0).reshape(-1)
    y = np.concatenate(y_all, axis=0).reshape(-1)
    return y, p


def pick_threshold_max_fpr(y, p, max_fpr=0.13):
    order = np.argsort(p)[::-1]
    p_sorted = p[order]
    y_sorted = y[order]

    n_ok = max(int((y_sorted == 0).sum()), 1)
    fp = 0
    best_thr = 1.0
    found = False

    for prob, yy in zip(p_sorted, y_sorted):
        if yy == 0:
            fp += 1
        fpr = fp / n_ok
        if fpr <= max_fpr:
            best_thr = prob
            found = True

    return float(best_thr), bool(found)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows):
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_csv(path: Path, row, fieldnames):
    ensure_dir(path.parent)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def metrics_at_threshold(y_true, p_def, thr):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    p_def = np.asarray(p_def).astype(float).reshape(-1)
    y_pred = (p_def >= float(thr)).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    fpr = fp / max(1, (fp + tn))

    return {
        "val_accuracy": float(acc),
        "val_precision": float(prec),
        "val_recall": float(rec),
        "val_f1": float(f1),
        "val_fpr": float(fpr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def write_error_analysis_json(path: Path, feature_names, X_val_scaled, y_true, p_def, thr, meta, top_k=50):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    p_def = np.asarray(p_def).astype(float).reshape(-1)
    y_pred = (p_def >= float(thr)).astype(int)

    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    fp_sorted = fp_idx[np.argsort(-p_def[fp_idx])] if fp_idx.size else fp_idx
    fn_sorted = fn_idx[np.argsort(p_def[fn_idx])] if fn_idx.size else fn_idx

    Xv = np.asarray(X_val_scaled, dtype=np.float32)

    def pack(i):
        x = Xv[i].reshape(-1).tolist()
        feats = {}
        for j, name in enumerate(feature_names):
            feats[name] = safe_float(x[j]) if j < len(x) else None
        return {
            "index": int(i),
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "prob_defect": float(p_def[i]),
            "features_scaled": feats,
        }

    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": float(thr),
        "summary": {
            "n_samples": int(len(y_true)),
            "n_fp": int(fp_idx.size),
            "n_fn": int(fn_idx.size),
            "n_tp": int(((y_true == 1) & (y_pred == 1)).sum()),
            "n_tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        },
        "top_false_positives": [pack(i) for i in fp_sorted[:top_k]],
        "top_false_negatives": [pack(i) for i in fn_sorted[:top_k]],
        "metadata": meta,
    }

    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def train_optimized(
    save_name="optimized_model.pt",
    scaler_rel="config/scaler.pkl",
    alpha=0.55,
    gamma=2.0,
    batch_size=64,
    lr=2e-4,
    weight_decay=1e-4,
    max_epochs=420,
    patience=45,
    max_fpr=0.13,
):
    root = get_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"root={root}")
    print(f"device={device}")

    scaler_path, feature_names, mean, scale = load_scaler(root, scaler_rel)
    (trX, trY), (vaX, vaY), (teX, teY) = _find_xy_splits(root)

    Xtr, ytr = _load_xy(trX, trY, feature_names, mean, scale)
    Xva, yva = _load_xy(vaX, vaY, feature_names, mean, scale)

    defect_rate_train = float(ytr.mean()) if len(ytr) else 0.0
    print(f"train_rows={len(ytr)} val_rows={len(yva)} defect_rate_train={defect_rate_train:.3f}")

    ytr_int = ytr.astype(int)
    class_counts = np.bincount(ytr_int, minlength=2).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    class_w = (class_counts.sum() / (2.0 * class_counts)).astype(np.float32)
    sample_w = class_w[ytr_int]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.float32),
        num_samples=len(sample_w),
        replacement=True,
    )

    model = MLP(len(feature_names)).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=5e-6
    )

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=batch_size, sampler=sampler, drop_last=False
    )
    va_loader = DataLoader(
        TensorDataset(torch.tensor(Xva), torch.tensor(yva)),
        batch_size=batch_size, shuffle=False, drop_last=False
    )

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    min_delta = 1e-4

    history_rows = []
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    for ep in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).view(-1)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        tr_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_auc, val_loss = eval_auc_loss(model, va_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"ep={ep:03d} loss={tr_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} lr={lr_now:.6f}")
        scheduler.step(val_auc)

        history_rows.append({
            "run_id": run_id,
            "epoch": int(ep),
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "val_auc": float(val_auc),
            "lr": float(lr_now),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
        })

        if val_auc > best_auc + min_delta:
            best_auc = val_auc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = ep
        best_auc = 0.0

    model.load_state_dict(best_state)
    y_val, p_val = get_probs(model, va_loader, device)
    thr_auto, found = pick_threshold_max_fpr(y_val.astype(int), p_val.astype(float), max_fpr=float(max_fpr))
    if not found:
        thr_auto = 0.60

    out_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / save_name

    torch.save(
        {
            "state_dict": best_state,
            "input_dim": len(feature_names),
            "best_auc": float(best_auc),
            "best_epoch": int(best_epoch),
            "feature_names": feature_names,
            "scaler_path": str(scaler_path),
            "threshold_auto": float(thr_auto),
            "threshold_rule": "max_fpr",
            "max_fpr": float(max_fpr),
            "loss_name": "focal",
            "focal_alpha": float(alpha),
            "focal_gamma": float(gamma),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "sampler": "weighted_random",
        },
        model_path,
    )

    print(f"Saved: {model_path}")
    print(f"Best AUC: {best_auc:.4f}  Best epoch: {best_epoch}")
    print(f"Auto threshold (val, max_fpr={max_fpr:.3f}): {thr_auto:.3f}  found={found}")

    results_dir = root / "results"
    ensure_dir(results_dir)

    training_history_path = results_dir / "training_history.csv"
    optimization_path = results_dir / "optimization_experiments.csv"
    error_analysis_path = results_dir / "error_analysis.json"

    write_csv(training_history_path, history_rows)

    thr_metrics = metrics_at_threshold(y_val, p_val, thr_auto)

    exp_fields = [
        "run_id", "created_at", "model_name",
        "epochs", "patience",
        "batch_size", "lr", "weight_decay",
        "alpha", "gamma", "max_fpr",
        "best_epoch", "best_auc",
        "pos_threshold", "threshold_found",
        "val_accuracy", "val_precision", "val_recall", "val_f1", "val_fpr",
        "tp", "tn", "fp", "fn",
        "notes",
    ]

    exp_row = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": "MLP",
        "epochs": int(max_epochs),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "max_fpr": float(max_fpr),
        "best_epoch": int(best_epoch),
        "best_auc": float(best_auc),
        "pos_threshold": float(thr_auto),
        "threshold_found": int(found),
        "val_accuracy": thr_metrics["val_accuracy"],
        "val_precision": thr_metrics["val_precision"],
        "val_recall": thr_metrics["val_recall"],
        "val_f1": thr_metrics["val_f1"],
        "val_fpr": thr_metrics["val_fpr"],
        "tp": thr_metrics["tp"],
        "tn": thr_metrics["tn"],
        "fp": thr_metrics["fp"],
        "fn": thr_metrics["fn"],
        "notes": "",
    }
    append_csv(optimization_path, exp_row, fieldnames=exp_fields)

    meta = {
        "run_id": run_id,
        "created_at": exp_row["created_at"],
        "save_name": save_name,
        "scaler_rel": scaler_rel,
        "scaler_path_resolved": str(scaler_path),
        "feature_count": int(len(feature_names)),
        "best_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "threshold_auto": float(thr_auto),
        "threshold_rule": "max_fpr",
        "max_fpr": float(max_fpr),
        "loss_name": "focal",
        "alpha": float(alpha),
        "gamma": float(gamma),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "sampler": "weighted_random",
        "device": str(device),
    }

    write_error_analysis_json(
        error_analysis_path,
        feature_names=feature_names,
        X_val_scaled=Xva,
        y_true=y_val,
        p_def=p_val,
        thr=thr_auto,
        meta=meta,
        top_k=50
    )

    print(f"Wrote: {training_history_path}")
    print(f"Appended: {optimization_path}")
    print(f"Wrote: {error_analysis_path}")


if __name__ == "__main__":
    train_optimized()
