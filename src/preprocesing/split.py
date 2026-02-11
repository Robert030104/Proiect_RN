from pathlib import Path
import numpy as np
import pandas as pd


def get_root() -> Path:
    return Path(__file__).resolve().parents[2]


def save_pair(out_dir: Path, x_name: str, y_name: str, X: pd.DataFrame, y: pd.Series):
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_dir / x_name, index=False)
    y.to_csv(out_dir / y_name, index=False)


def stratified_split(df, y_col="defect", seed=42, test_size=0.15, val_size=0.15):
    rng = np.random.default_rng(seed)

    idx0 = df.index[df[y_col] == 0].to_numpy()
    idx1 = df.index[df[y_col] == 1].to_numpy()
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def split_idx(idx):
        n = len(idx)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        test = idx[:n_test]
        val = idx[n_test:n_test + n_val]
        train = idx[n_test + n_val:]
        return train, val, test

    tr0, va0, te0 = split_idx(idx0)
    tr1, va1, te1 = split_idx(idx1)

    train_idx = np.concatenate([tr0, tr1])
    val_idx = np.concatenate([va0, va1])
    test_idx = np.concatenate([te0, te1])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def main():
    root = get_root()
    in_path = root / "data" / "processed" / "dataset_processed.csv"
    df = pd.read_csv(in_path)

    y = df["defect"].astype(int)
    X = df.drop(columns=["defect"])

    tr_idx, va_idx, te_idx = stratified_split(df, y_col="defect", seed=42, test_size=0.15, val_size=0.15)

    X_train, y_train = X.loc[tr_idx].reset_index(drop=True), y.loc[tr_idx].reset_index(drop=True)
    X_val, y_val = X.loc[va_idx].reset_index(drop=True), y.loc[va_idx].reset_index(drop=True)
    X_test, y_test = X.loc[te_idx].reset_index(drop=True), y.loc[te_idx].reset_index(drop=True)

    save_pair(root / "data" / "train", "X_train.csv", "y_train.csv", X_train, y_train)
    save_pair(root / "data" / "validation", "X_val.csv", "y_val.csv", X_val, y_val)
    save_pair(root / "data" / "test", "X_test.csv", "y_test.csv", X_test, y_test)

    print(f"train: rows={len(y_train)} defect_rate={float(y_train.mean()):.3f}")
    print(f"val:   rows={len(y_val)} defect_rate={float(y_val.mean()):.3f}")
    print(f"test:  rows={len(y_test)} defect_rate={float(y_test.mean()):.3f}")


if __name__ == "__main__":
    main()
