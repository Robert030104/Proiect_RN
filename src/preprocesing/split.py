# src/preprocesing/split.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def get_root():
    return Path(__file__).resolve().parents[2]


def split():
    root = get_root()
    proc_path = root / "data" / "processed" / "dataset_processed.csv"
    if not proc_path.exists():
        raise FileNotFoundError(f"Nu exista: {proc_path}")

    df = pd.read_csv(proc_path)
    if "defect" not in df.columns:
        raise ValueError("Lipseste coloana 'defect' in dataset_processed.csv")

    y = df["defect"].astype(int)
    X = df.drop(columns=["defect"])

    # 70% train, 30% temp (val+test)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 15% val, 15% test (din total)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    def save_split(name, Xp, yp):
        d = root / "data" / name
        d.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: numele exacte pe care le cauta train_model.py
        if name == "train":
            x_path = d / "X_train.csv"
            y_path = d / "y_train.csv"
        elif name == "validation":
            x_path = d / "X_val.csv"
            y_path = d / "y_val.csv"
        elif name == "test":
            x_path = d / "X_test.csv"
            y_path = d / "y_test.csv"
        else:
            x_path = d / f"X_{name}.csv"
            y_path = d / f"y_{name}.csv"

        Xp.to_csv(x_path, index=False)
        yp.to_csv(y_path, index=False)

        print(f"Saved: {x_path} (cols={Xp.shape[1]})")
        print(f"Saved: {y_path}")

    save_split("train", X_tr, y_tr)
    save_split("validation", X_val, y_val)
    save_split("test", X_te, y_te)

    print(f"Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}")
    print(f"Rates: {y_tr.mean():.3f} / {y_val.mean():.3f} / {y_te.mean():.3f}")
    print(f"Features saved: {X.shape[1]}")


if __name__ == "__main__":
    split()
