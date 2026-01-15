import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

RAW_DATA_PATH = "data/raw/dataset_auto.csv"
TRAIN_PATH = "data/train/"
VAL_PATH = "data/validation/"
TEST_PATH = "data/test/"
SCALER_PATH = "config/scaler.pkl"

os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(VAL_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)
os.makedirs("config", exist_ok=True)

data = pd.read_csv(RAW_DATA_PATH)

COLS = [
    "km",
    "vechime_ani",
    "temperatura_motor",
    "temperatura_ulei",
    "presiune_ulei",
    "vibratii",
    "ore_de_la_revizie",
    "km_de_la_schimb_ulei",
    "maf",
    "map_val",
]

X = data[COLS]
y = data["defect"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
scaler.fit(X_train.values)
joblib.dump(scaler, SCALER_PATH)

# salvam BRUT + header
X_train.to_csv(TRAIN_PATH + "X_train.csv", index=False)
pd.Series(y_train).to_csv(TRAIN_PATH + "y_train.csv", index=False)

X_val.to_csv(VAL_PATH + "X_val.csv", index=False)
pd.Series(y_val).to_csv(VAL_PATH + "y_val.csv", index=False)

X_test.to_csv(TEST_PATH + "X_test.csv", index=False)
pd.Series(y_test).to_csv(TEST_PATH + "y_test.csv", index=False)

print("Preprocesarea datelor a fost finalizata cu succes (X brut + scaler.pkl).")
