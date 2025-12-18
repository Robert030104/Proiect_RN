import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model import DefectPredictor
from sklearn.metrics import recall_score

# --------------------
# reproducibilitate
# --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# paths
# --------------------
X_TRAIN_PATH = "data/train/X_train.csv"
Y_TRAIN_PATH = "data/train/y_train.csv"
X_VAL_PATH = "data/validation/X_val.csv"
Y_VAL_PATH = "data/validation/y_val.csv"
MODEL_PATH = "models/model_predictie_defecte.pth"

# --------------------
# hiperparametri
# --------------------
EPOCHS = 50
LR = 0.001
BATCH_SIZE = 32

# --------------------
# load data
# --------------------
def load_xy(x_path, y_path):
    X = torch.tensor(pd.read_csv(x_path).values, dtype=torch.float32)
    y = torch.tensor(pd.read_csv(y_path).values, dtype=torch.long).squeeze()
    return X, y

X_train, y_train = load_xy(X_TRAIN_PATH, Y_TRAIN_PATH)
X_val, y_val = load_xy(X_VAL_PATH, Y_VAL_PATH)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

input_dim = X_train.shape[1]

# --------------------
# model
# --------------------
model = DefectPredictor(input_dim).to(device)

# ⚠️ CLASĂ DEFECT MAI IMPORTANTĂ
class_weights = torch.tensor([1.0, 3.0], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=LR)

# --------------------
# training loop
# --------------------
best_recall = 0.0
os.makedirs("models", exist_ok=True)

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

    # --------------------
    # validation
    # --------------------
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

    # --------------------
    # save best model
    # --------------------
    if val_recall > best_recall:
        best_recall = val_recall
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\nCel mai bun model salvat (Recall DEFECT = {best_recall*100:.2f}%)")
