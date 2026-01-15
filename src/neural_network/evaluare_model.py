import sys
from pathlib import Path

import torch
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

def find_project_root(start: Path) -> Path:
    p = start
    for _ in range(10):
        if (p / "config").exists() and (p / "models").exists() and (p / "data").exists():
            return p
        p = p.parent
    return start

HERE = Path(__file__).resolve().parent
ROOT = find_project_root(HERE)

# incearca sa importe modelul din src/neural_network
sys.path.insert(0, str(ROOT / "src" / "neural_network"))
from model import DefectPredictor

X_test_path = ROOT / "data/test/X_test.csv"
y_test_path = ROOT / "data/test/y_test.csv"

X_test_df = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).values.ravel()

scaler = joblib.load(ROOT / "config/scaler.pkl")
X_test = scaler.transform(X_test_df.values)

model = DefectPredictor(X_test.shape[1])
model.load_state_dict(torch.load(ROOT / "models/model_predictie_defecte.pth", map_location="cpu"))
model.eval()

X_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    outputs = model(X_tensor)
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print("\nProcente:")
    print(f"Recall DEFECT (TP/(TP+FN)): {tp / (tp + fn) * 100:.2f}%")
    print(f"False Negative rate (FN/(TP+FN)): {fn / (tp + fn) * 100:.2f}%")
    print(f"False Positive rate (FP/(FP+TN)): {fp / (fp + tn) * 100:.2f}%")

print("\nRaport clasificare:")
print(classification_report(y_test, y_pred, target_names=["NORMAL", "DEFECT"], digits=4, zero_division=0))
