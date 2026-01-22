# src/app/app.py
# Streamlit app - Predictie defect auto (V5)
# Ruleaza din root: streamlit run src/app/app.py

from pathlib import Path
import sys
import pickle
import numpy as np
import streamlit as st
import torch

# ---------- PATH FIX (ca sa mearga importul indiferent cum rulezi) ----------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.neural_network.model import MLP


# -------------------- Loaders --------------------

def get_root() -> Path:
    return ROOT


@st.cache_resource
def load_scaler(root: Path):
    p = root / "scaler.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Lipseste scaler.pkl: {p}")
    with open(p, "rb") as f:
        sc = pickle.load(f)

    feature_names = list(sc["feature_names"])
    mean = np.array(sc["mean"], dtype=np.float32)
    scale = np.array(sc["scale"], dtype=np.float32)
    return feature_names, mean, scale, p


@st.cache_resource
def load_model(root: Path):
    model_path = root / "models" / "model_predictie_defecte.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Lipseste modelul: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    input_dim = int(ckpt["input_dim"])

    model = MLP(input_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --- FIX AICI ---
    thr_raw = ckpt.get("threshold_rule", ckpt.get("threshold_auto", 0.60))
    try:
        thr = float(thr_raw)
    except Exception:
        # daca e string gen "max_fpr", folosim threshold_auto
        thr = float(ckpt.get("threshold_auto", 0.60))

    max_fpr = float(ckpt.get("max_fpr", 0.13))
    pos_mult = float(ckpt.get("pos_mult", 1.0))
    loss_name = str(ckpt.get("loss_name", "bce"))

    return model, thr, max_fpr, pos_mult, loss_name, model_path



# -------------------- Helpers --------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def compute_derived(raw: dict):
    """
    Calculeaza automat feature-uri V5 daca sunt disponibile:
    - delta_temp
    - flags: pressure_low, pressure_very_low, service_overdue, coolant_overheat, oil_overheat,
             battery_low, vibratii_high, airflow_mismatch
    - pressure_expected/residual/margin
    - anomaly_count
    - temp_stress, pressure_stress, health_index
    """
    ct = raw.get("coolant_temp_c")
    ot = raw.get("oil_temp_c")
    op = raw.get("oil_pressure_bar")
    maf = raw.get("maf_gps")
    mapk = raw.get("map_kpa")
    bv = raw.get("battery_v")
    vib = raw.get("vibratii_relanti")
    zile = raw.get("zile_de_la_ultima_revizie")
    age = raw.get("vechime_ani")

    # delta_temp
    if ct is not None and ot is not None:
        raw["delta_temp"] = float(ot - ct)

    # service_overdue
    if zile is not None:
        raw["service_overdue"] = int(float(zile) >= 180)

    # pressure flags
    if op is not None:
        raw["pressure_low"] = int(float(op) < 1.20)
        raw["pressure_very_low"] = int(float(op) < 0.90)

    # overheat flags
    if ct is not None:
        raw["coolant_overheat"] = int(float(ct) > 102.0)
    if ot is not None:
        raw["oil_overheat"] = int(float(ot) > 115.0)

    # battery / vibration flags
    if bv is not None:
        raw["battery_low"] = int(float(bv) < 12.2)
    if vib is not None:
        raw["vibratii_high"] = int(float(vib) > 3.2)

    # airflow mismatch
    if maf is not None and mapk is not None:
        maf_expected = 2.6 + 0.20 * float(mapk)
        raw["airflow_mismatch"] = int((float(maf) < (maf_expected - 2.2)) and (float(mapk) > 55.0))

    # pressure_expected/residual/margin
    if op is not None:
        raw["pressure_margin"] = float(op) - 1.20

    if op is not None and ot is not None and mapk is not None and age is not None:
        pressure_expected = (
            3.0
            - 0.020 * (float(ot) - 95.0)
            + 0.004 * (float(mapk) - 35.0)
            - 0.020 * (float(age) - 6.0)
        )
        raw["pressure_expected"] = float(pressure_expected)
        raw["pressure_residual"] = float(float(op) - pressure_expected)

    # anomaly_count
    keys = [
        "coolant_overheat", "oil_overheat",
        "pressure_low", "pressure_very_low",
        "battery_low", "vibratii_high",
        "airflow_mismatch", "service_overdue"
    ]
    if all(k in raw for k in keys):
        raw["anomaly_count"] = int(sum(int(raw[k]) for k in keys))

    # temp_stress
    if "delta_temp" in raw and ct is not None and ot is not None:
        dt = float(raw["delta_temp"])
        raw["temp_stress"] = float(
            max(0.0, float(ct) - 98.0) / 20.0
            + max(0.0, float(ot) - 110.0) / 25.0
            + max(0.0, dt - 18.0) / 18.0
        )

    # pressure_stress
    if op is not None and "pressure_residual" in raw:
        pr = float(raw["pressure_residual"])
        raw["pressure_stress"] = float(
            max(0.0, 1.30 - float(op)) / 1.30
            + max(0.0, -pr) / 1.20
        )

    # health_index 0..1 (soft, orientativ)
    if "temp_stress" in raw and "pressure_stress" in raw:
        ts = float(raw["temp_stress"])
        ps = float(raw["pressure_stress"])
        so = float(raw.get("service_overdue", 0))
        am = float(raw.get("airflow_mismatch", 0))
        bl = float(raw.get("battery_low", 0))

        vib_score = 0.0
        if vib is not None:
            vib_score = 0.18 * ((float(vib) - 1.5) / 3.0)

        base = 0.42 * ts + 0.55 * ps + vib_score + 0.12 * so + 0.10 * am + 0.08 * bl
        base = clamp(base, 0.0, 4.0)
        raw["health_index"] = float(base / 4.0)

    return raw


def compute_trends(curr: dict, prev: dict, delta_days: float, delta_km: float):
    """
    Trenduri V5 (normalizate similar cu dataset):
      - trend_temp = (ct1-ct0)/5
      - trend_pressure = (op0-op1)/0.35 (scadere presiune => pozitiv)
      - trend_vibratii = (vib1-vib0)/0.35
      - trend_battery = (batt0-batt1)/0.60 (scadere baterie => pozitiv)
    """
    if "coolant_temp_c" in curr and "coolant_temp_c" in prev:
        curr["trend_temp"] = clamp((float(curr["coolant_temp_c"]) - float(prev["coolant_temp_c"])) / 5.0, -1.0, 2.0)

    if "oil_pressure_bar" in curr and "oil_pressure_bar" in prev:
        curr["trend_pressure"] = clamp((float(prev["oil_pressure_bar"]) - float(curr["oil_pressure_bar"])) / 0.35, -1.0, 2.0)

    if "vibratii_relanti" in curr and "vibratii_relanti" in prev:
        curr["trend_vibratii"] = clamp((float(curr["vibratii_relanti"]) - float(prev["vibratii_relanti"])) / 0.35, -1.0, 2.0)

    if "battery_v" in curr and "battery_v" in prev:
        curr["trend_battery"] = clamp((float(prev["battery_v"]) - float(curr["battery_v"])) / 0.60, -1.0, 2.0)

    curr["delta_days"] = int(delta_days)
    curr["delta_km"] = int(delta_km)
    return curr


def estimate_km_remaining(prob_defect: float, critical: bool):
    if critical:
        return 0, 500, "CRITIC: risc mare. Recomandat stop/diagnoza imediata."
    p = prob_defect
    if p < 0.20:
        return 12000, 25000, "Risc mic. Monitorizare si revizie normala."
    if p < 0.35:
        return 7000, 15000, "Risc moderat-scazut. Verificari de rutina recomandate."
    if p < 0.50:
        return 3000, 9000, "Risc moderat. Recomandat control in curand."
    if p < 0.65:
        return 1000, 5000, "Risc ridicat. Programare la service cat mai curand."
    return 0, 2000, "Risc foarte ridicat. Diagnoza rapida recomandata."


def rule_recommendations(raw: dict):
    rec = []
    critical = False

    ct = raw.get("coolant_temp_c")
    ot = raw.get("oil_temp_c")
    op = raw.get("oil_pressure_bar")
    bv = raw.get("battery_v")
    vib = raw.get("vibratii_relanti")
    zile = raw.get("zile_de_la_ultima_revizie")
    maf = raw.get("maf_gps")
    mapk = raw.get("map_kpa")

    if ct is not None and float(ct) >= 110:
        rec.append("Verifica nivelul antigel/apa, radiatorul, termostatul si ventilatorul (temperatura lichid racire foarte mare).")
        critical = True

    if ot is not None and float(ot) >= 125:
        rec.append("Verifica uleiul (nivel/vascozitate), racirea uleiului si posibile frecari interne (temperatura ulei foarte mare).")
        critical = True

    if op is not None and float(op) <= 0.90:
        rec.append("Presiune ulei foarte mica: daca e real, opreste motorul. Verifica pompa/filtru/sorb/uzuri.")
        critical = True
    elif op is not None and float(op) < 1.20:
        rec.append("Presiune ulei scazuta: verifica nivel ulei, filtru, posibile pierderi/uzura.")

    if bv is not None and float(bv) < 12.0:
        rec.append("Baterie joasa: verifica alternatorul, bornele, bateria, consumatori paraziti.")

    if vib is not None and float(vib) > 3.5:
        rec.append("Vibratii mari la relanti: verifica suporti motor, injectie/aprindere, admisie, echilibrare.")

    if zile is not None and float(zile) >= 240:
        rec.append("Revizie depasita: ulei/filtre/inspectie (risc crescut pe termen scurt).")

    if maf is not None and mapk is not None and float(mapk) > 55:
        maf_expected = 2.6 + 0.20 * float(mapk)
        if float(maf) < (maf_expected - 2.2):
            rec.append("Debit aer suspect (MAF mic la sarcina): verifica filtru aer, admisie, MAF, EGR/boost leaks.")

    return critical, rec


# -------------------- UI --------------------

st.set_page_config(page_title="Predictie defect auto (RN) - V5", page_icon="🚗", layout="centered")
st.title("🚗 Predictie defect auto (RN) — V5")
st.caption("Model demonstrativ pe date simulate. Rezultatul este orientativ, nu garantie.")

root = get_root()

try:
    feature_names, mean, scale, scaler_path = load_scaler(root)
    model, thr, max_fpr, pos_mult, loss_name, model_path = load_model(root)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.expander("Detalii model (informativ)", expanded=False):
    st.write(f"Model: `{model_path}`")
    st.write(f"Scaler: `{scaler_path}`")
    st.write(f"Features: **{len(feature_names)}**")
    st.write(f"Prag (rule/max_fpr): **{thr:.3f}**  | max_fpr setat: **{max_fpr:.3f}**")
    st.write(f"Loss: **{loss_name}** | pos_mult: **{pos_mult:.2f}**")

st.subheader("Masurare curenta (t1)")

c1, c2 = st.columns(2)
with c1:
    kilometraj_total = st.number_input("Kilometraj total", min_value=0, value=150000, step=1000)
    vechime_ani = st.number_input("Vechime (ani)", min_value=0, value=8, step=1)
    zile_revizie = st.number_input("Zile de la ultima revizie", min_value=0, value=120, step=10)
    map_kpa = st.number_input("MAP (kPa)", min_value=0.0, value=45.0, step=1.0, format="%.2f")
    maf_gps = st.number_input("MAF (g/s)", min_value=0.0, value=11.0, step=0.1, format="%.2f")

with c2:
    coolant_temp_c = st.number_input("Temp. lichid racire (°C)", min_value=0.0, value=92.0, step=0.5, format="%.2f")
    oil_temp_c = st.number_input("Temp. ulei (°C)", min_value=0.0, value=98.0, step=0.5, format="%.2f")
    oil_pressure_bar = st.number_input("Presiune ulei (bar)", min_value=0.0, value=2.4, step=0.05, format="%.3f")
    battery_v = st.number_input("Baterie (V)", min_value=0.0, value=13.8, step=0.05, format="%.2f")
    vibratii_relanti = st.number_input("Vibratii relanti (rel)", min_value=0.0, value=2.0, step=0.05, format="%.2f")

st.subheader("Optional: masurare anterioara (t0) pentru trend-uri V5")
use_prev = st.toggle("Am masurare anterioara (t0)", value=False)

prev = {}
delta_days = 0
delta_km = 0

if use_prev:
    p1, p2 = st.columns(2)
    with p1:
        prev["coolant_temp_c"] = st.number_input("t0: Temp. lichid racire (°C)", min_value=0.0, value=90.0, step=0.5, format="%.2f")
        prev["oil_pressure_bar"] = st.number_input("t0: Presiune ulei (bar)", min_value=0.0, value=2.6, step=0.05, format="%.3f")
        delta_days = st.number_input("Zile intre masurari", min_value=1, value=60, step=1)
    with p2:
        prev["vibratii_relanti"] = st.number_input("t0: Vibratii relanti", min_value=0.0, value=1.8, step=0.05, format="%.2f")
        prev["battery_v"] = st.number_input("t0: Baterie (V)", min_value=0.0, value=13.9, step=0.05, format="%.2f")
        delta_km = st.number_input("Km intre masurari", min_value=0, value=2500, step=100)

st.divider()

raw = {
    "kilometraj_total": float(kilometraj_total),
    "vechime_ani": float(vechime_ani),
    "coolant_temp_c": float(coolant_temp_c),
    "oil_temp_c": float(oil_temp_c),
    "oil_pressure_bar": float(oil_pressure_bar),
    "maf_gps": float(maf_gps),
    "map_kpa": float(map_kpa),
    "battery_v": float(battery_v),
    "vibratii_relanti": float(vibratii_relanti),
    "zile_de_la_ultima_revizie": float(zile_revizie),
}

if use_prev:
    raw = compute_trends(raw, prev, float(delta_days), float(delta_km))

raw = compute_derived(raw)

# completeaza ce lipseste cu 0 (siguranta)
missing = [f for f in feature_names if f not in raw]
for f in missing:
    raw[f] = 0.0

# avertizare daca modelul asteapta trenduri, dar user nu a dat t0
needs_trend = any(f.startswith("trend_") for f in feature_names) or any(f in feature_names for f in ["delta_days", "delta_km"])
if needs_trend and not use_prev:
    st.warning("Modelul V5 foloseste trend-uri (t0->t1). Daca nu introduci masurarea anterioara, "
               "predictia poate fi mai putin precisa (trend-urile vor fi 0).")

X = np.array([raw[f] for f in feature_names], dtype=np.float32).reshape(1, -1)
X_scaled = (X - mean.reshape(1, -1)) / (scale.reshape(1, -1) + 1e-12)

with torch.no_grad():
    logits = model(torch.tensor(X_scaled, dtype=torch.float32))
    prob = float(torch.sigmoid(logits).cpu().numpy().reshape(-1)[0])

pred_defect = prob >= float(thr)

critical, recs = rule_recommendations(raw)
min_km, max_km, km_text = estimate_km_remaining(prob, critical)

st.subheader("📊 Rezultat")
st.metric("Probabilitate defect", f"{prob*100:.1f}%")
st.write(f"**Predictie:** {'❌ DEFECT PROBABIL' if pred_defect else '✅ OK'}  (prag: {thr:.3f})")
st.write(f"**Estimare km ramasi (orientativ):** {min_km} – {max_km} km")
st.caption(km_text)

st.subheader("🔧 Recomandari (pe baza semnalelor)")
if recs:
    for r in recs:
        st.write(f"- {r}")
else:
    st.write("Nu exista semnale clare pe reguli. Daca apar simptome, fa o diagnoza OBD si verificari de rutina.")

with st.expander("Date folosite (debug)", expanded=False):
    keys_show = [
        "kilometraj_total","vechime_ani","coolant_temp_c","oil_temp_c","oil_pressure_bar",
        "maf_gps","map_kpa","battery_v","vibratii_relanti","zile_de_la_ultima_revizie",
        "delta_temp","service_overdue","pressure_low","pressure_very_low",
        "coolant_overheat","oil_overheat","battery_low","vibratii_high","airflow_mismatch",
        "pressure_expected","pressure_residual","temp_stress","pressure_stress","health_index",
        "delta_days","delta_km","trend_temp","trend_pressure","trend_vibratii","trend_battery","anomaly_count"
    ]
    st.json({k: raw.get(k, None) for k in keys_show if k in raw})
