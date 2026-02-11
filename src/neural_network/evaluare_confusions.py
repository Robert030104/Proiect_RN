import argparse
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import matplotlib.pyplot as plt

from model import MLP


def get_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_scaler_dict(path: Path):
    with open(path, "rb") as f:
        sc = pickle.load(f)
    feature_names = list(sc["feature_names"])
    mean = np.array(sc["mean"], dtype=np.float32)
    scale = np.array(sc["scale"], dtype=np.float32)
    return feature_names, mean, scale


def read_y_csv(path: Path):
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


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        return state, meta
    if isinstance(ckpt, dict):
        return ckpt, {}
    return ckpt, {}


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


def pick_threshold_from_meta(meta: dict):
    if isinstance(meta, dict):
        if "threshold_f1" in meta:
            return float(meta["threshold_f1"]), "threshold_f1"
        if "threshold_auto" in meta:
            return float(meta["threshold_auto"]), "threshold_auto"
    return 0.5, "default_0.5"


def plot_confusion(cm, out_path: Path, title: str):
    plt.figure(figsize=(6.4, 5.2))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Normal", "Defect"], rotation=0)
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


def compute_metrics(y_true, p, thr):
    y_pred = (p >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--optimized", required=True)
    ap.add_argument("--trained", default=None)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--thr_optimized", type=float, default=None)
    ap.add_argument("--thr_trained", type=float, default=None)
    args = ap.parse_args()

    root = get_root()
    x_path = (root / args.x).resolve() if not Path(args.x).is_absolute() else Path(args.x)
    y_path = (root / args.y).resolve() if not Path(args.y).is_absolute() else Path(args.y)
    scaler_path = (root / args.scaler).resolve() if not Path(args.scaler).is_absolute() else Path(args.scaler)

    feature_names, mean, scale = load_scaler_dict(scaler_path)
    X = preprocess_X(x_path, feature_names, mean, scale)
    y_true = read_y_csv(y_path)

    out_shots = root / "docs" / "screenshots"
    out_res = root / "results"
    out_res.mkdir(parents=True, exist_ok=True)

    final = {"test_file_x": str(x_path), "test_file_y": str(y_path), "models": {}}

    state_o, meta_o = load_checkpoint(args.optimized)
    input_dim_o = int(meta_o.get("input_dim", X.shape[1]))
    model_o = MLP(input_dim_o)
    model_o.load_state_dict(state_o, strict=True)
    p_o = predict_proba(model_o, X)

    if args.thr_optimized is not None:
        thr_o = float(args.thr_optimized)
        thr_src_o = "cli"
    else:
        thr_o, thr_src_o = pick_threshold_from_meta(meta_o)

    m_o = compute_metrics(y_true, p_o, thr_o)
    plot_confusion(m_o["confusion_matrix"], out_shots / "confusion_matrix_optimized.png", "Confusion Matrix - Optimized")
    final["models"]["optimized"] = {
        "path": args.optimized,
        "threshold_used": thr_o,
        "threshold_source": thr_src_o,
        "accuracy": m_o["accuracy"],
        "f1": m_o["f1"],
        "precision": m_o["precision"],
        "recall": m_o["recall"],
        "auc": m_o["auc"],
        "confusion_matrix": m_o["confusion_matrix"].tolist(),
    }

    if args.trained:
        state_t, meta_t = load_checkpoint(args.trained)
        input_dim_t = int(meta_t.get("input_dim", X.shape[1]))
        model_t = MLP(input_dim_t)
        model_t.load_state_dict(state_t, strict=True)
        p_t = predict_proba(model_t, X)

        if args.thr_trained is not None:
            thr_t = float(args.thr_trained)
            thr_src_t = "cli"
        else:
            thr_t, thr_src_t = pick_threshold_from_meta(meta_t)

        m_t = compute_metrics(y_true, p_t, thr_t)
        plot_confusion(m_t["confusion_matrix"], out_shots / "confusion_matrix_baseline.png", "Confusion Matrix - Trained")
        final["models"]["trained"] = {
            "path": args.trained,
            "threshold_used": thr_t,
            "threshold_source": thr_src_t,
            "accuracy": m_t["accuracy"],
            "f1": m_t["f1"],
            "precision": m_t["precision"],
            "recall": m_t["recall"],
            "auc": m_t["auc"],
            "confusion_matrix": m_t["confusion_matrix"].tolist(),
        }

    out_json = out_res / "final_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print(f"Saved: {out_shots / 'confusion_matrix_optimized.png'}")
    if args.trained:
        print(f"Saved: {out_shots / 'confusion_matrix_baseline.png'}")
    print(f"Saved: {out_json}")
    print(f"optimized: acc={final['models']['optimized']['accuracy']:.4f} f1={final['models']['optimized']['f1']:.4f} auc={final['models']['optimized']['auc']:.4f} thr={thr_o:.3f} ({thr_src_o})")


if __name__ == "__main__":
    main()
