from pathlib import Path
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_intercept(raw_score, target, iters=90):
    lo, hi = -30.0, 30.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        m = float(sigmoid(raw_score + mid).mean())
        if m < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def generate_dataset(n=12000, defect_target=0.25, seed=42, label_noise=0.0):
    rng = np.random.default_rng(seed)

    vechime_ani = rng.integers(1, 21, size=n).astype(np.float32)
    km_mean = 12000.0 * vechime_ani + rng.normal(0, 26000, size=n)
    kilometraj_total = np.clip(km_mean, 5000, 420000).astype(np.float32)

    zile_de_la_ultima_revizie = rng.integers(0, 420, size=n).astype(np.float32)
    service_overdue = (zile_de_la_ultima_revizie >= 180).astype(int).astype(np.float32)

    map_kpa = np.clip(rng.normal(42, 14, size=n), 25, 95).astype(np.float32)

    # Baseline (t0)
    coolant_t0 = (
        86.0
        + 0.20 * (map_kpa - 35.0)
        + 0.12 * (vechime_ani - 6.0)
        + rng.normal(0, 2.8, size=n)
    ).astype(np.float32)

    oil_t0 = (
        coolant_t0
        + 4.5
        + 0.12 * (map_kpa - 35.0)
        + rng.normal(0, 2.2, size=n)
    ).astype(np.float32)

    battery_t0 = (
        14.10
        - 0.055 * (vechime_ani - 6.0)
        + rng.normal(0, 0.20, size=n)
    ).astype(np.float32)

    vibr_t0 = (
        1.30
        + 0.070 * (vechime_ani - 4.0)
        + 0.0000032 * kilometraj_total
        + rng.normal(0, 0.30, size=n)
    ).astype(np.float32)

    maf_expected0 = 2.6 + 0.20 * map_kpa
    maf_t0 = (maf_expected0 + rng.normal(0, 1.4, size=n)).astype(np.float32)

    oilp_t0 = (
        3.10
        - 0.022 * (oil_t0 - 95.0)
        + 0.004 * (map_kpa - 35.0)
        - 0.030 * (vechime_ani - 6.0)
        - 0.0000022 * (kilometraj_total - 120000.0)
        + rng.normal(0, 0.18, size=n)
    ).astype(np.float32)

    coolant_t0 = np.clip(coolant_t0, 70, 125)
    oil_t0 = np.clip(oil_t0, 75, 140)
    oilp_t0 = np.clip(oilp_t0, 0.60, 4.2)
    maf_t0 = np.clip(maf_t0, 1.5, 25.0)
    battery_t0 = np.clip(battery_t0, 10.5, 14.8)
    vibr_t0 = np.clip(vibr_t0, 0.8, 6.0)

    # Fault propensities (latent) - depend on age/km/service
    km_norm = (kilometraj_total - 180000.0) / 120000.0
    age_norm = (vechime_ani - 8.0) / 6.0

    p_cooling = sigmoid(-4.0 + 1.0 * age_norm + 0.9 * km_norm + 0.9 * service_overdue)
    p_lube = sigmoid(-3.9 + 1.1 * age_norm + 1.0 * km_norm + 0.8 * service_overdue)
    p_air = sigmoid(-4.3 + 0.9 * age_norm + 0.8 * km_norm + 0.25 * (map_kpa > 55).astype(np.float32))
    p_elec = sigmoid(-4.6 + 1.2 * age_norm + 0.9 * km_norm)

    cooling_fault = (rng.random(n) < p_cooling).astype(np.float32)
    lube_fault = (rng.random(n) < p_lube).astype(np.float32)
    air_fault = (rng.random(n) < p_air).astype(np.float32)
    elec_fault = (rng.random(n) < p_elec).astype(np.float32)

    # Time gap
    delta_days = rng.integers(30, 121, size=n).astype(np.float32)
    delta_km = np.clip(rng.normal(2200, 900, size=n), 500, 6000).astype(np.float32)

    # Trends - IMPORTANT: make them truly predictive
    # Cooling -> temp rises
    trend_temp_raw = (
        rng.normal(0.2, 0.6, size=n)
        + cooling_fault * rng.uniform(2.0, 6.0, size=n)
        + lube_fault * rng.uniform(0.6, 2.0, size=n)
        + 0.25 * service_overdue
    ).astype(np.float32)

    # Lube -> pressure drops
    trend_pressure_raw = (
        rng.normal(0.02, 0.05, size=n)
        + lube_fault * rng.uniform(0.18, 0.45, size=n)
        + 0.03 * service_overdue
    ).astype(np.float32)

    # Vibration rises
    trend_vibr_raw = (
        rng.normal(0.02, 0.05, size=n)
        + lube_fault * rng.uniform(0.10, 0.28, size=n)
        + elec_fault * rng.uniform(0.03, 0.12, size=n)
        + 0.03 * (delta_km / 3000.0)
    ).astype(np.float32)

    # Battery drops
    trend_batt_raw = (
        rng.normal(0.00, 0.06, size=n)
        + elec_fault * rng.uniform(0.15, 0.45, size=n)
        + 0.04 * (vechime_ani / 10.0)
    ).astype(np.float32)

    # Airflow degrades when air_fault under load
    trend_air_raw = (
        rng.normal(0.00, 0.40, size=n)
        + air_fault * rng.uniform(1.2, 3.2, size=n) * (map_kpa > 55).astype(np.float32)
    ).astype(np.float32)

    # t1 values
    coolant = coolant_t0 + trend_temp_raw + cooling_fault * rng.uniform(2, 8, size=n).astype(np.float32)
    oil = oil_t0 + 0.6 * trend_temp_raw + lube_fault * rng.uniform(2, 9, size=n).astype(np.float32)
    oilp = oilp_t0 - trend_pressure_raw - lube_fault * rng.uniform(0.3, 1.0, size=n).astype(np.float32)
    vibr = vibr_t0 + trend_vibr_raw
    batt = battery_t0 - trend_batt_raw
    maf = maf_t0 - trend_air_raw

    coolant_temp_c = np.clip(coolant, 70, 134).astype(np.float32)
    oil_temp_c = np.clip(oil, 75, 150).astype(np.float32)
    oil_pressure_bar = np.clip(oilp, 0.35, 4.2).astype(np.float32)
    vibratii_relanti = np.clip(vibr, 0.8, 7.0).astype(np.float32)
    battery_v = np.clip(batt, 9.3, 14.8).astype(np.float32)
    maf_gps = np.clip(maf, 1.2, 25.0).astype(np.float32)

    delta_temp = (oil_temp_c - coolant_temp_c).astype(np.float32)

    pressure_low = (oil_pressure_bar < 1.20).astype(int)
    pressure_very_low = (oil_pressure_bar < 0.90).astype(int)

    coolant_overheat = (coolant_temp_c > 102.0).astype(int)
    oil_overheat = (oil_temp_c > 115.0).astype(int)

    battery_low = (battery_v < 12.2).astype(int)
    vibratii_high = (vibratii_relanti > 3.2).astype(int)

    maf_expected = 2.6 + 0.20 * map_kpa
    airflow_mismatch = ((maf_gps < (maf_expected - 2.2)) & (map_kpa > 55)).astype(int)

    pressure_margin = (oil_pressure_bar - 1.20).astype(np.float32)

    pressure_expected = (
        3.0
        - 0.020 * (oil_temp_c - 95.0)
        + 0.004 * (map_kpa - 35.0)
        - 0.020 * (vechime_ani - 6.0)
    ).astype(np.float32)

    pressure_residual = (oil_pressure_bar - pressure_expected).astype(np.float32)

    # Trend features normalized (like before)
    trend_temp = np.clip(trend_temp_raw / 5.0, -1.0, 2.0).astype(np.float32)
    trend_pressure = np.clip(trend_pressure_raw / 0.35, -1.0, 2.0).astype(np.float32)
    trend_vibratii = np.clip(trend_vibr_raw / 0.35, -1.0, 2.0).astype(np.float32)
    trend_battery = np.clip(trend_batt_raw / 0.60, -1.0, 2.0).astype(np.float32)

    anomaly_count = (
        coolant_overheat
        + oil_overheat
        + pressure_low
        + pressure_very_low
        + battery_low
        + vibratii_high
        + airflow_mismatch
        + service_overdue.astype(int)
    ).astype(np.float32)

    temp_stress = (
        np.maximum(0.0, coolant_temp_c - 98.0) / 20.0
        + np.maximum(0.0, oil_temp_c - 110.0) / 25.0
        + np.maximum(0.0, delta_temp - 18.0) / 18.0
    ).astype(np.float32)

    pressure_stress = (
        np.maximum(0.0, 1.30 - oil_pressure_bar) / 1.30
        + np.maximum(0.0, -pressure_residual) / 1.20
    ).astype(np.float32)

    # Health index (0..1)
    health_index = (
        0.42 * temp_stress
        + 0.55 * pressure_stress
        + 0.22 * trend_temp
        + 0.35 * trend_pressure
        + 0.18 * trend_vibratii
        + 0.12 * trend_battery
        + 0.18 * (anomaly_count / 8.0)
        + rng.normal(0, 0.06, size=n).astype(np.float32)
    ).astype(np.float32)
    health_index = np.clip(health_index, 0.0, 4.0).astype(np.float32)
    health_index = (health_index / 4.0).astype(np.float32)

    # ===== Label (DEFECT) - now strongly tied to trend/stress =====
    # This is the critical fix vs your weak V5.
    raw_score = (
        0.8 * cooling_fault
        + 1.0 * lube_fault
        + 0.6 * air_fault
        + 0.6 * elec_fault
        + 2.2 * health_index
        + 0.60 * temp_stress
        + 0.90 * pressure_stress
        + 0.50 * (anomaly_count / 8.0)
        + 0.55 * np.maximum(0.0, trend_pressure)   # pressure drop is very predictive
        + 0.35 * np.maximum(0.0, trend_temp)
        + rng.normal(0, 0.18, size=n).astype(np.float32)
    ).astype(np.float32)

    b = calibrate_intercept(raw_score, defect_target, iters=90)
    p_def = sigmoid(raw_score + b).astype(np.float32)
    defect = (rng.random(n) < p_def).astype(np.int32)

    if label_noise and float(label_noise) > 0:
        flip = (rng.random(n) < float(label_noise))
        defect = np.where(flip, 1 - defect, defect).astype(np.int32)

    kilometraj_total_t1 = np.clip(kilometraj_total + delta_km, 5000, 450000).astype(np.float32)

    df = pd.DataFrame(
        {
            "kilometraj_total": np.round(kilometraj_total_t1, 0).astype(int),
            "vechime_ani": np.round(vechime_ani, 0).astype(int),
            "coolant_temp_c": np.round(coolant_temp_c, 2),
            "oil_temp_c": np.round(oil_temp_c, 2),
            "oil_pressure_bar": np.round(oil_pressure_bar, 3),
            "maf_gps": np.round(maf_gps, 2),
            "map_kpa": np.round(map_kpa, 2),
            "battery_v": np.round(battery_v, 2),
            "vibratii_relanti": np.round(vibratii_relanti, 2),
            "zile_de_la_ultima_revizie": np.round(zile_de_la_ultima_revizie, 0).astype(int),
            "delta_temp": np.round(delta_temp, 2),
            "pressure_low": pressure_low.astype(int),
            "pressure_very_low": pressure_very_low.astype(int),
            "service_overdue": service_overdue.astype(int),
            "coolant_overheat": coolant_overheat.astype(int),
            "oil_overheat": oil_overheat.astype(int),
            "battery_low": battery_low.astype(int),
            "vibratii_high": vibratii_high.astype(int),
            "airflow_mismatch": airflow_mismatch.astype(int),
            "pressure_margin": np.round(pressure_margin, 3),
            "pressure_expected": np.round(pressure_expected, 3),
            "pressure_residual": np.round(pressure_residual, 3),

            "delta_days": np.round(delta_days, 0).astype(int),
            "delta_km": np.round(delta_km, 0).astype(int),
            "trend_temp": np.round(trend_temp, 4),
            "trend_pressure": np.round(trend_pressure, 4),
            "trend_vibratii": np.round(trend_vibratii, 4),
            "trend_battery": np.round(trend_battery, 4),
            "anomaly_count": np.round(anomaly_count, 0).astype(int),
            "temp_stress": np.round(temp_stress, 4),
            "pressure_stress": np.round(pressure_stress, 4),
            "health_index": np.round(health_index, 4),

            "defect": defect.astype(int),
        }
    )

    return df


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "data" / "raw" / "dataset_auto.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(n=12000, defect_target=0.25, seed=42, label_noise=0.0)
    df.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"randuri={len(df)} defect_rate={df['defect'].mean()*100:.2f}%")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
