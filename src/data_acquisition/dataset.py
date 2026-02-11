from pathlib import Path
import random
import numpy as np
import pandas as pd


def get_root() -> Path:
    return Path(__file__).resolve().parents[2]


def clip(x, lo, hi):
    return float(max(lo, min(hi, x)))


def gen_normal(rng):
    km = rng.uniform(5_000, 220_000)
    vechime_ani = rng.uniform(0.2, 14.0)
    zile_de_la_ultima_revizie = rng.uniform(0, 240)

    coolant_temp = rng.normal(88.0, 6.0)
    oil_temp = rng.normal(95.0, 8.0)
    oil_pressure = rng.normal(2.4, 0.35)

    maf = rng.normal(7.0, 1.6)
    map_kpa = rng.normal(55.0, 10.0)
    battery_v = rng.normal(12.5, 0.35)

    vibratii_relanti = rng.normal(1.2, 0.45)

    coolant_temp = clip(coolant_temp, 70, 102)
    oil_temp = clip(oil_temp, 75, 112)
    oil_pressure = clip(oil_pressure, 1.7, 3.4)

    maf = clip(maf, 2.5, 14.0)
    map_kpa = clip(map_kpa, 25.0, 90.0)
    battery_v = clip(battery_v, 11.8, 13.5)

    vibratii_relanti = clip(vibratii_relanti, 0.3, 2.4)

    return {
        "km": float(km),
        "vechime_ani": float(vechime_ani),
        "zile_de_la_ultima_revizie": float(zile_de_la_ultima_revizie),
        "coolant_temp": float(coolant_temp),
        "oil_temp": float(oil_temp),
        "oil_pressure": float(oil_pressure),
        "maf": float(maf),
        "map_kpa": float(map_kpa),
        "battery_v": float(battery_v),
        "vibratii_relanti": float(vibratii_relanti),
    }


def inject_defect_signature(row, rng):
    defect_type = rng.choice(["supraincalzire", "presiune_ulei", "vibratii", "baterie_slaba", "mixt"])

    if defect_type == "supraincalzire":
        row["coolant_temp"] = clip(row["coolant_temp"] + rng.uniform(10, 22), 85, 125)
        row["oil_temp"] = clip(row["oil_temp"] + rng.uniform(12, 24), 90, 135)
        row["oil_pressure"] = clip(row["oil_pressure"] - rng.uniform(0.1, 0.35), 0.7, 3.4)

    elif defect_type == "presiune_ulei":
        row["oil_pressure"] = clip(row["oil_pressure"] - rng.uniform(0.6, 1.2), 0.6, 3.4)
        row["oil_temp"] = clip(row["oil_temp"] + rng.uniform(4, 12), 75, 135)

    elif defect_type == "vibratii":
        row["vibratii_relanti"] = clip(row["vibratii_relanti"] + rng.uniform(2.2, 5.5), 0.3, 10.0)
        row["maf"] = clip(row["maf"] + rng.uniform(-1.5, 2.5), 2.5, 16.0)

    elif defect_type == "baterie_slaba":
        row["battery_v"] = clip(row["battery_v"] - rng.uniform(0.8, 1.4), 9.8, 13.5)
        row["map_kpa"] = clip(row["map_kpa"] + rng.uniform(-6, 10), 20.0, 95.0)

    else:
        row["oil_pressure"] = clip(row["oil_pressure"] - rng.uniform(0.5, 1.0), 0.6, 3.4)
        row["coolant_temp"] = clip(row["coolant_temp"] + rng.uniform(8, 18), 80, 125)
        row["vibratii_relanti"] = clip(row["vibratii_relanti"] + rng.uniform(1.5, 4.0), 0.3, 10.0)

    row["km"] = clip(row["km"] + rng.uniform(20_000, 90_000), 0, 350_000)
    row["zile_de_la_ultima_revizie"] = clip(row["zile_de_la_ultima_revizie"] + rng.uniform(60, 240), 0, 400)
    row["vechime_ani"] = clip(row["vechime_ani"] + rng.uniform(0.3, 2.5), 0.1, 20.0)

    return defect_type


def score_risk(row):
    r = 0.0
    r += max(0.0, (row["coolant_temp"] - 95.0) / 20.0)
    r += max(0.0, (row["oil_temp"] - 110.0) / 25.0)
    r += max(0.0, (1.6 - row["oil_pressure"]) / 0.8)
    r += max(0.0, (row["vibratii_relanti"] - 2.2) / 3.5)
    r += max(0.0, (11.9 - row["battery_v"]) / 1.0)
    r += max(0.0, (row["km"] - 180_000) / 130_000)
    r += max(0.0, (row["zile_de_la_ultima_revizie"] - 180) / 160)
    return r


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def enforce_separation(row, label, rng):
    if label == 1:
        bad = 0
        if row["oil_pressure"] <= 1.35:
            bad += 1
        if row["coolant_temp"] >= 100.0:
            bad += 1
        if row["oil_temp"] >= 115.0:
            bad += 1
        if row["vibratii_relanti"] >= 3.2:
            bad += 1
        if row["battery_v"] <= 11.6:
            bad += 1

        if bad < 2:
            picks = rng.choice(["oil_pressure", "coolant_temp", "oil_temp", "vibratii_relanti"], size=2, replace=False)
            for f in picks:
                if f == "oil_pressure":
                    row["oil_pressure"] = clip(row["oil_pressure"] - rng.uniform(0.6, 1.0), 0.6, 3.4)
                elif f == "coolant_temp":
                    row["coolant_temp"] = clip(row["coolant_temp"] + rng.uniform(10, 18), 80, 125)
                elif f == "oil_temp":
                    row["oil_temp"] = clip(row["oil_temp"] + rng.uniform(10, 18), 80, 135)
                elif f == "vibratii_relanti":
                    row["vibratii_relanti"] = clip(row["vibratii_relanti"] + rng.uniform(2.0, 4.5), 0.3, 10.0)

    else:
        row["coolant_temp"] = clip(row["coolant_temp"], 70, 97.0)
        row["oil_temp"] = clip(row["oil_temp"], 75, 112.0)
        row["vibratii_relanti"] = clip(row["vibratii_relanti"], 0.3, 2.6)
        row["oil_pressure"] = clip(row["oil_pressure"], 1.7, 3.4)
        row["battery_v"] = clip(row["battery_v"], 11.8, 13.5)

    return row


def generate(n_rows=12000, defect_target=0.25, seed=42, label_noise=0.02):
    rng = np.random.default_rng(seed)
    rows = []

    base_offset = 1.38
    for _ in range(n_rows):
        row = gen_normal(rng)

        risk = score_risk(row)
        p = logistic(risk - base_offset)

        p = 0.75 * p + 0.25 * defect_target
        p = float(clip(p, 0.01, 0.99))

        label = 1 if rng.random() < p else 0

        defect_type = "none"
        if label == 1:
            defect_type = inject_defect_signature(row, rng)

        row = enforce_separation(row, label, rng)

        if rng.random() < label_noise:
            label = 1 - label
            if label == 0:
                defect_type = "none"

        row["defect"] = int(label)
        row["defect_type"] = defect_type
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    root = get_root()
    out_dir = root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_auto.csv"

    df = generate(n_rows=12000, defect_target=0.25, seed=42, label_noise=0.02)
    df.to_csv(out_path, index=False)
    rate = float(df["defect"].mean())
    print(f"Saved: {out_path}")
    print(f"rows={len(df)} defect_rate={rate:.3f}")


if __name__ == "__main__":
    main()
