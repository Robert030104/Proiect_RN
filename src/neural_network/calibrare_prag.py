import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# paths (ruleaza din oriunde)
ROOT = Path(__file__).resolve().parents[2]  # D:\Proiect_RN\Proiect_RN

VAL_X_PATH = ROOT / "data" / "validation" / "X_val.csv"
VAL_Y_PATH = ROOT / "data" / "validation" / "y_val.csv"
SCALER_PATH = ROOT / "config" / "scaler.pkl"
MODEL_PATH = ROOT / "models" / "model_predictie_defecte.pth"

# import model (src/neural_network/model.py)
import sys
sys.path.append(str(ROOT / "src"))
from neural_network.model import DefectPredictor


def load_val_data():
    X_val = pd.read_csv(VAL_X_PATH).values
    y_val = pd.read_csv(VAL_Y_PATH).values.ravel().astype(int)
    return X_val, y_val


def predict_proba_defect(model, X_scaled):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        out = model(X_tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()
    return probs[:, 1]  # prob defect


def main():
    print("=== Calibrare prag decizie (pe VALIDATION) ===")
    print(f"ROOT: {ROOT}")
    print(f"Model: {MODEL_PATH}")
    print(f"Scaler: {SCALER_PATH}")
    print(f"X_val: {VAL_X_PATH}")
    print(f"y_val: {VAL_Y_PATH}")

    # load
    X_val, y_val = load_val_data()
    scaler = joblib.load(SCALER_PATH)

    # aplica scaler (daca X_val e deja scalat, comenteaza linia urmatoare)
    X_val_scaled = scaler.transform(X_val)

    # model
    input_dim = X_val_scaled.shape[1]
    model = DefectPredictor(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # probs
    p_def = predict_proba_defect(model, X_val_scaled)

    # sweep thresholds
    thresholds = np.arange(0.05, 0.96, 0.05)

    best_acc = -1.0
    best_thr_acc = None
    best_stats_acc = None

    best_f1 = -1.0
    best_thr_f1 = None
    best_stats_f1 = None

    print("\nthr   acc     f1_macro  recall_defect  fp_rate   fn_rate")
    print("----------------------------------------------------------")

    for thr in thresholds:
        y_pred = (p_def >= thr).astype(int)

        acc = accuracy_score(y_val, y_pred)
        f1m = f1_score(y_val, y_pred, average="macro", zero_division=0)
        rec_def = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        print(f"{thr:0.2f}  {acc:0.4f}  {f1m:0.4f}   {rec_def:0.4f}       {fp_rate:0.4f}   {fn_rate:0.4f}")

        if acc > best_acc:
            best_acc = acc
            best_thr_acc = thr
            best_stats_acc = (acc, f1m, rec_def, fp_rate, fn_rate, cm)

        if f1m > best_f1:
            best_f1 = f1m
            best_thr_f1 = thr
            best_stats_f1 = (acc, f1m, rec_def, fp_rate, fn_rate, cm)

    # results
    print("\n=== REZULTAT (max accuracy) ===")
    acc, f1m, rec_def, fp_rate, fn_rate, cm = best_stats_acc
    print(f"Best threshold (accuracy): {best_thr_acc:0.2f}")
    print(f"Accuracy: {acc*100:.2f}% | F1 macro: {f1m:.4f} | Recall DEFECT: {rec_def*100:.2f}%")
    print(f"FP rate: {fp_rate*100:.2f}% | FN rate: {fn_rate*100:.2f}%")
    print("Confusion Matrix [ [TN FP], [FN TP] ]:")
    print(cm)

    print("\n=== REZULTAT (max F1 macro) ===")
    acc, f1m, rec_def, fp_rate, fn_rate, cm = best_stats_f1
    print(f"Best threshold (F1 macro): {best_thr_f1:0.2f}")
    print(f"Accuracy: {acc*100:.2f}% | F1 macro: {f1m:.4f} | Recall DEFECT: {rec_def*100:.2f}%")
    print(f"FP rate: {fp_rate*100:.2f}% | FN rate: {fn_rate*100:.2f}%")
    print("Confusion Matrix [ [TN FP], [FN TP] ]:")
    print(cm)

    print("\nPune in app.py:")
    print(f"threshold = {best_thr_acc:0.2f}  # (sau {best_thr_f1:0.2f} daca vrei F1 maxim)")


if __name__ == "__main__":
    main()
