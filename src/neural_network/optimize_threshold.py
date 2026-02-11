import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from model import MLP


def get_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_y_csv(path):
    df = pd.read_csv(path)
    if "defect" in df.columns:
        y = pd.to_numeric(df["defect"], errors="coerce").fillna(0).astype(int).values
    else:
        y = pd.to_numeric(df.iloc[:, 0], errors="coerce").fillna(0).astype(int).values
    return np.where(y >= 0.5, 1, 0).astype(int)


def load_scaler_dict(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Lipseste scaler: {path}")
    with open(path, "rb") as f:
        sc = pickle.load(f)
    feature_names = list(sc["feature_names"])
    mean = np.array(sc["mean"], dtype=np.float32)
    scale = np.array(sc["scale"], dtype=np.float32)
    return feature_names, mean, scale


def preprocess_X(X_df, feature_names, mean, scale):
    missing = [c for c in feature_names if c not in X_df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane in X: {missing}")

    X = X_df[feature_names].copy()
    for c in feature_names:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    X = X.values.astype(np.float32)
    X = (X - mean) / (scale + 1e-12)
    return X


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt
    return {"state_dict": ckpt}


@torch.no_grad()
def predict_proba(model, X_np, batch_size=512):
    X = torch.tensor(X_np, dtype=torch.float32)
    probs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0).reshape(-1)


def main():
    root = get_root()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)

    ap.add_argument(
        "--scaler",
        default=str(root / "config" / "scaler.pkl"),
        help="default: root/config/scaler.pkl"
    )

    ap.add_argument("--min_acc", type=float, default=0.75)
    ap.add_argument("--out_key", default="threshold_f1")
    args = ap.parse_args()

    model_path = Path(args.model)
    model_path = model_path if model_path.is_absolute() else (root / model_path)
    model_path = model_path.resolve()

    x_path = Path(args.x)
    x_path = x_path if x_path.is_absolute() else (root / x_path)
    x_path = x_path.resolve()

    y_path = Path(args.y)
    y_path = y_path if y_path.is_absolute() else (root / y_path)
    y_path = y_path.resolve()

    scaler_path = Path(args.scaler)
    scaler_path = scaler_path if scaler_path.is_absolute() else (root / scaler_path)
    scaler_path = scaler_path.resolve()

    X_df = pd.read_csv(x_path)
    y = read_y_csv(y_path)

    ckpt = load_checkpoint(model_path)
    feature_names, mean, scale = load_scaler_dict(scaler_path)
    X = preprocess_X(X_df, feature_names, mean, scale)

    input_dim = int(ckpt.get("input_dim", X.shape[1]))
    model = MLP(input_dim)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    p = predict_proba(model, X)

    best = None
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (p >= thr).astype(int)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, zero_division=0))
        if acc < args.min_acc:
            continue
        cand = (f1, acc, float(thr))
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand

    if best is None:
        print("No threshold meets min_acc. Try lowering min_acc or improving model.")
        return

    f1, acc, thr = best
    print(f"best_thr={thr:.3f} f1={f1:.4f} acc={acc:.4f}")
    print(f"scaler_used={scaler_path}")

    ckpt[args.out_key] = float(thr)
    ckpt["scaler_path"] = str(scaler_path)
    torch.save(ckpt, model_path)
    print(f"updated checkpoint key: {args.out_key}")


if __name__ == "__main__":
    main()
