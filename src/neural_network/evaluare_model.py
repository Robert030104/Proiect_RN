import torch
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from model import DefectPredictor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# load data
X_test = pd.read_csv(ROOT / "data/test/X_test.csv")
y_test = pd.read_csv(ROOT / "data/test/y_test.csv").values.ravel()

# load scaler
scaler = joblib.load(ROOT / "config/scaler.pkl")
X_test = scaler.transform(X_test)

# load model
model = DefectPredictor(X_test.shape[1])
model.load_state_dict(torch.load(ROOT / "models/model_predictie_defecte.pth", map_location="cpu"))
model.eval()

# predict
X_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    outputs = model(X_tensor)
    y_pred = torch.argmax(outputs, dim=1).numpy()

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(cm)

print("\nProcente:")
print(f"True Positive (DEFECT detectat): {tp / (tp + fn) * 100:.2f}%")
print(f"False Negative (DEFECT ratat): {fn / (tp + fn) * 100:.2f}%")
print(f"False Positive (alarme false): {fp / (fp + tn) * 100:.2f}%")

print("\nRaport clasificare:")
print(classification_report(y_test, y_pred, target_names=["NORMAL", "DEFECT"]))
