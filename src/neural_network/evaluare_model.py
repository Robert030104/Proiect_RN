from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

from model import MLP


def get_root():
    return Path(__file__).resolve().parents[2]


def load_scaler(root: Path):
    sp = root / "scaler.pkl"
    if not sp.exists():
        raise FileNotFoundError(f"Lipseste: {sp}")
    with open(sp, "rb") as f:
        sc = pickle.load(f)
    feat = list(sc["feature_names"])
    mean = np.array(sc["mean"], dtype=np.float32)
    scale = np.array(sc["scale"], dtype=np.float32)
    return feat, mean, scale


def load_test_xy(root: Path, feat, mean, scale):
    x_path = root / "data" / "test" / "X_test.csv"
    y_path = root / "data" / "test" / "y_test.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"Lipseste: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Lipseste: {y_path}")

    Xdf = pd.read_csv(x_path)
    ydf = pd.read_csv(y_path)

    if "defect" in ydf.columns:
        y = pd.to_numeric(ydf["defect"], errors="coerce").fillna(0).astype(int).values
    else:
        y = pd.to_numeric(ydf.iloc[:, 0], errors="coerce").fillna(0).astype(int).values

    missing = [c for c in feat if c not in Xdf.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane in {x_path}: {missing}")

    X = Xdf[feat].copy()
    for c in feat:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    X = X.values.astype(np.float32)
    X = (X - mean) / (scale + 1e-12)
    return X, y


def report_at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / max(len(y), 1)
    precision = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)

    fpr = fp / max((fp + tn), 1)
    tnr = tn / max((tn + fp), 1)
    fnr = fn / max((fn + tp), 1)

    rep = classification_report(y, pred, target_names=["OK", "Defect"], digits=4)
    return tn, fp, fn, tp, acc, precision, recall, fpr, tnr, fnr, rep


def evaluate():
    root = get_root()
    model_path = root / "models" / "model_predictie_defecte.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Lipseste: {model_path}")

    feat, mean, scale = load_scaler(root)
    X, y = load_test_xy(root, feat, mean, scale)

    ckpt = torch.load(model_path, map_location="cpu")
    model = MLP(int(ckpt["input_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        p = torch.sigmoid(logits).numpy().reshape(-1)

    roc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    pr = average_precision_score(y, p) if len(np.unique(y)) > 1 else 0.0

    thr = float(ckpt.get("threshold_auto", 0.60))
    rule = ckpt.get("threshold_rule", "fixed")
    max_fpr = float(ckpt.get("max_fpr", 0.11))

    tn, fp, fn, tp, acc, prec, rec, fpr, tnr, fnr, rep = report_at_threshold(y, p, thr)

    print("=== EVALUARE TEST (clar) ===")
    print(f"Model: {model_path}")
    print(f"Test samples: {len(y)} | Defect rate: {y.mean()*100:.2f}%")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC:  {pr:.4f}")

    loss_name = ckpt.get("loss_name", "bce")
    print(f"Loss: {loss_name} pos_mult={ckpt.get('pos_mult', 'n/a')}")

    print()
    print("--- Prag folosit ---")
    if rule == "max_fpr":
        print(f"regula: max_fpr={max_fpr:.3f}")
    else:
        print("regula: fixed")
    print(f"prag probabilitate: {thr:.3f}")
    print()
    print("--- Confusion matrix ---")
    print(f"TN (OK prezis OK):        {tn}")
    print(f"FP (OK prezis Defect):    {fp}   <- alarme false")
    print(f"FN (Defect prezis OK):    {fn}   <- defecte ratate")
    print(f"TP (Defect prezis Defect):{tp}")
    print()
    print("--- Metrici usor de inteles ---")
    print(f"Accuracy:   {acc:.3f}")
    print(f"Precision (Defect): {prec:.3f}")
    print(f"Recall (Defect):    {rec:.3f}")
    print(f"FPR:        {fpr:.3f}")
    print(f"TNR:        {tnr:.3f}")
    print(f"FNR:        {fnr:.3f}")
    print(f"Alarme false la 100 OK: {fpr*100:.1f}")
    print()
    print("--- Raport clasificare ---")
    print(rep)


if __name__ == "__main__":
    evaluate()
