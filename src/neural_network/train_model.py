import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score
from pathlib import Path
import sys

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# radacina proiectului
HERE = Path(__file__).resolve()
# daca train_model.py e in src/neural_network, root e cu 2 nivele sus
ROOT = HERE.parents[2] if (HERE.parents[2] / "config").exists() else HERE.parent

# permite importul modelului
sys.path.insert(0, str(HERE.parent))
from model import DefectPredictor  # model.py in acelasi folder cu train_model.py

X_TRAIN_PATH = ROOT / "data/train/X_train.csv"
Y_TRAIN_PATH = ROOT / "data/train/y_train.csv"
X_VAL_PATH = ROOT / "data/validation/X_val.csv"
Y_VAL_PATH = ROOT / "data/validation/y_val.csv"
MODEL_PATH = ROOT / "models/model_predictie_defecte.pth"
SCALER_PATH = ROOT / "config/scaler.pkl"

EPOCHS = 50
LR = 0.001
BATCH_SIZE = 32

scaler = joblib.load(SCALER_PATH)

def load_xy(x_path, y_path):
    X_np = pd.read_csv(x_path).values
    X_np = scaler.transform(X_np)  # <-- scalare aici (o singura data)
    X = torch.tensor(X_np, dtype=torch.float32)

    y = pd.read_csv(y_path).values.squeeze()
    y = torch.tensor(y, dtype=torch.long)
    return X, y

X_train, y_train = load_xy(X_TRAIN_PATH, Y_TRAIN_PATH)
X_val, y_val = load_xy(X_VAL_PATH, Y_VAL_PATH)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

input_dim = X_train.shape[1]
model = DefectPredictor(input_dim).to(device)

# DEFECT mai important (ca sa nu prezica doar NORMAL)
class_weights = torch.tensor([1.0, 10.0], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=LR)

best_recall = 0.0
os.makedirs(ROOT / "models", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        outputs = model(X_val.to(device))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        val_recall = recall_score(y_val.numpy(), preds, zero_division=0)

    train_loss /= len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Recall DEFECT (val): {val_recall*100:.2f}%"
    )

    if val_recall > best_recall:
        best_recall = val_recall
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\nCel mai bun model salvat (Recall DEFECT = {best_recall*100:.2f}%)")
