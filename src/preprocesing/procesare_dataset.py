from pathlib import Path
import pickle
import numpy as np
import pandas as pd


def get_root():
    return Path(__file__).resolve().parents[2]


def preprocess():
    root = get_root()

    raw_path = root / "data" / "raw" / "dataset_auto.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Nu exista: {raw_path}")

    df = pd.read_csv(raw_path)
    if "defect" not in df.columns:
        raise ValueError("Lipseste coloana 'defect'")

    y = pd.to_numeric(df["defect"], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=["defect"]).copy()

    # fortam numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # imputare
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    feature_names = list(X.columns)

    Xv = X.values.astype(np.float32)
    mean = Xv.mean(axis=0).astype(np.float32)
    std = Xv.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    # IMPORTANT: salvam PROCESSED ca RAW (nescalat)
    out_df = X.copy()
    out_df["defect"] = y.values.astype(int)

    proc_dir = root / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_csv = proc_dir / "dataset_processed.csv"
    out_df.to_csv(out_csv, index=False)

    scaler_path = root / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(
            {
                "feature_names": feature_names,
                "mean": mean,
                "scale": std,
            },
            f,
        )

    print(f"OK: {raw_path}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {scaler_path}")
    print(f"Rows: {len(df)}  Defect_rate: {y.mean()*100:.2f}%")
    print(f"Features: {len(feature_names)}")


if __name__ == "__main__":
    preprocess()
