import os
import torch
import pandas as pd
from model import DefectPredictor

os.makedirs("models", exist_ok=True)

X_train = pd.read_csv("data/train/X_train.csv")
input_dim = X_train.shape[1]

model = DefectPredictor(input_dim)

torch.save(model.state_dict(), "models/model_predictie_defecte.pth")

print("Model RN definit si salvat (neantrenat).")
