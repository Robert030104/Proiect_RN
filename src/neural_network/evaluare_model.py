import argparse
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt

from model import MLP


def get_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_scaler(path: Path):
    with open(path, "rb") as f:
        sc = pickle.load(f)
    feature_names = list(sc["feature_names"])
    mean = np.array(sc["mean"], dtype=np.float32)
    scale = np.array(sc["scale"], dtype=np.float32)
    return feature_names, mean, scale


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        return state, meta
    if isinstance(ckpt, dict):
        return ckpt, {}
    return ckpt, {}


def read_y(path: Path):
    df = pd.read_csv(path)
    if "defect" in df.columns:
        y = pd.to_numeric(df["defect"], errors="coerce").fillna(0).astype(int).values
    else:
        y = pd.to_numeric(df.iloc[:, 0], errors="coerce").fillna(0).astype(int).values
    y = np.where(y >= 0.5, 1, 0).astype(int)
    return y


def preprocess_X(x_path: Path, feature_names, mean, scale):
    Xdf = pd.read_csv(x_path)

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
    return X


@torch.no_grad()
def predict_proba(model, X_np, batch_size=1024):
    model.eval()
    X = torch.tensor(X_np, dtype=torch.float32)
    probs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0).reshape(-1)


def pick_threshold(meta: dict, fallback=0.5):
    if "threshold_f1" in meta:
        return float(meta["threshold_f1"]), "threshold_f1"
    if "threshold_auto" in meta:
        return float(meta["threshold_auto"]), "threshold_auto"
    return float(fallback), "default"


def save_confusion_png(cm, out_path: Path, title: str):
    plt.figure(figsize=(6.4, 5.2))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Normal", "Defect"])
    plt.yticks(ticks, ["Normal", "Defect"])

    thresh = cm.max() * 0.55 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            plt.text(j, i, f"{val}", ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/optimized_model.pt")
    ap.add_argument("--x", default="data/test/X_test.csv")
    ap.add_argument("--y", default="data/test/y_test.csv")
    ap.add_argument("--scaler", default="scaler.pkl")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--save_cm", action="store_true")
    args = ap.parse_args()

    root = get_root()

    model_path = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    x_path = (root / args.x).resolve() if not Path(args.x).is_absolute() else Path(args.x)
    y_path = (root / args.y).resolve() if not Path(args.y).is_absolute() else Path(args.y)
    scaler_path = (root / args.scaler).resolve() if not Path(args.scaler).is_absolute() else Path(args.scaler)

    feature_names, mean, scale = load_scaler(scaler_path)
    X = preprocess_X(x_path, feature_names, mean, scale)
    y_true = read_y(y_path)

    state, meta = load_checkpoint(model_path)
    input_dim = int(meta.get("input_dim", len(feature_names)))
    if input_dim != len(feature_names):
        input_dim = len(feature_names)

    model = MLP(input_dim)
    model.load_state_dict(state, strict=True)

    p = predict_proba(model, X)

    if args.threshold is not None:
        thr = float(args.threshold)
        thr_src = "cli"
    else:
        thr, thr_src = pick_threshold(meta, fallback=0.5)

    y_pred = (p >= thr).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else 0.5
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    out_res = root / "results"
    out_res.mkdir(parents=True, exist_ok=True)
    out_json = out_res / "final_metrics.json"

    payload = {
        "model_path": str(model_path),
        "x_path": str(x_path),
        "y_path": str(y_path),
        "threshold_used": thr,
        "threshold_source": thr_src,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "feature_names": feature_names,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.save_cm:
        out_cm = root / "docs" / "screenshots" / "confusion_matrix_optimized.png"
        save_confusion_png(cm, out_cm, "Confusion Matrix - Optimized")

    print(f"acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} thr={thr:.3f} ({thr_src})")
    print(f"Saved: {out_json}")
    if args.save_cm:
        print(f"Saved: {root / 'docs' / 'screenshots' / 'confusion_matrix_optimized.png'}")


if __name__ == "__main__":
    main()
