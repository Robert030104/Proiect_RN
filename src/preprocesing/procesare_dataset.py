from pathlib import Path
import pickle
import numpy as np
import pandas as pd


def get_root() -> Path:
    return Path(__file__).resolve().parents[2]


FEATURES = [
    "km",
    "vechime_ani",
    "zile_de_la_ultima_revizie",
    "coolant_temp",
    "oil_temp",
    "oil_pressure",
    "maf",
    "map_kpa",
    "battery_v",
    "vibratii_relanti",
]


def main():
    root = get_root()
    in_path = root / "data" / "raw" / "dataset_auto.csv"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_processed.csv"

    df = pd.read_csv(in_path)

    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["defect"] = pd.to_numeric(df["defect"], errors="coerce").fillna(0).astype(int)
    df["defect"] = (df["defect"] >= 1).astype(int)

    df = df.dropna(subset=FEATURES).copy()
    df = df.reset_index(drop=True)

    X = df[FEATURES].values.astype(np.float32)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    scaler = {
        "feature_names": FEATURES,
        "mean": mean.tolist(),
        "scale": std.tolist(),
    }

    scaler_path = root / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    keep_cols = FEATURES + ["defect"]
    df[keep_cols].to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Saved: {scaler_path}")
    print(f"rows={len(df)} defect_rate={float(df['defect'].mean()):.3f}")


if __name__ == "__main__":
    main()
