import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from model import MLP


def read_y_csv(path):
    df = pd.read_csv(path)
    if "defect" in df.columns:
        y = pd.to_numeric(df["defect"], errors="coerce").fillna(0).astype(int).values
    else:
        y = pd.to_numeric(df.iloc[:, 0], errors="coerce").fillna(0).astype(int).values
    return np.where(y >= 0.5, 1, 0).astype(int)


def load_scaler_dict(path):
    with open(path, "rb") as f:
        sc = pickle.load(f)
    feature_names = list(sc["feature_names"])
    mean = np.array(sc["mean"], dtype=np.float32)
    scale = np.array(sc["scale"], dtype=np.float32)
    return feature_names, mean, scale


def preprocess_X(X_df, feature_names, mean, scale):
    X = X_df[feature_names].copy()
    for c in feature_names:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
    X = X.values.astype(np.float32)
    X = (X - mean) / (scale + 1e-12)
    return X


def load_checkpoint(path):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--min_acc", type=float, default=0.75)
    ap.add_argument("--out_key", default="threshold_f1")
    args = ap.parse_args()

    X_df = pd.read_csv(args.x)
    y = read_y_csv(args.y)

    ckpt = load_checkpoint(args.model)
    feature_names, mean, scale = load_scaler_dict(args.scaler)
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

    ckpt[args.out_key] = float(thr)
    torch.save(ckpt, args.model)
    print(f"updated checkpoint key: {args.out_key}")


if __name__ == "__main__":
    main()
