import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# paths
RAW_DATA_PATH = "data/raw/dataset_auto.csv"
TRAIN_PATH = "data/train/"
VAL_PATH = "data/validation/"
TEST_PATH = "data/test/"
PROCESSED_PATH = "data/processed/"
SCALER_PATH = "config/scaler.pkl"

# create folders if they do not exist
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(VAL_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

# 1. Incarca datele brute
data = pd.read_csv(RAW_DATA_PATH)

X = data.drop("defect", axis=1).values
y = data["defect"].values

# 2. Split 70% / 15% / 15%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 3. Standardizare
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 4. Salveza scaler-ul
joblib.dump(scaler, SCALER_PATH)

# 5. Salveaza datasetile preprocesate
pd.DataFrame(X_train).to_csv(TRAIN_PATH + "X_train.csv", index=False)
pd.DataFrame(y_train).to_csv(TRAIN_PATH + "y_train.csv", index=False)

pd.DataFrame(X_val).to_csv(VAL_PATH + "X_val.csv", index=False)
pd.DataFrame(y_val).to_csv(VAL_PATH + "y_val.csv", index=False)

pd.DataFrame(X_test).to_csv(TEST_PATH + "X_test.csv", index=False)
pd.DataFrame(y_test).to_csv(TEST_PATH + "y_test.csv", index=False)

pd.DataFrame(data).to_csv(PROCESSED_PATH + "dataset_clean.csv", index=False)

print("Preprocesarea datelor a fost finalizata cu succes!")


