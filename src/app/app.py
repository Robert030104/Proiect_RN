from pathlib import Path
import sys
import json
import pickle
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.neural_network.model import MLP


def get_root() -> Path:
    return ROOT


@st.cache_resource
def load_scaler(root: Path):
    p = root / "config" / "scaler.pkl"
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
    model_path = root / "models" / "optimized_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Lipseste modelul optimizat: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError("Checkpoint invalid: lipseste cheia 'state_dict'.")

    input_dim = int(ckpt.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError("Checkpoint invalid: input_dim lipseste/invalid.")

    model = MLP(input_dim)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    if "threshold_f1" in ckpt:
        thr = float(ckpt["threshold_f1"])
        thr_src = "threshold_f1"
    elif "threshold_auto" in ckpt:
        thr = float(ckpt["threshold_auto"])
        thr_src = "threshold_auto"
    else:
        thr = 0.5
        thr_src = "default_0.5"

    meta = {
        "thr": thr,
        "thr_src": thr_src,
        "max_fpr": float(ckpt.get("max_fpr", 0.13)),
        "loss_name": str(ckpt.get("loss_name", "bce")),
        "best_auc": float(ckpt.get("best_auc", -1.0)),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
    }

    return model, meta, model_path


@st.cache_resource
def load_final_metrics(root: Path):
    p = root / "results" / "final_metrics.json"
    if not p.exists():
        return None, p
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, p


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def compute_derived(raw: dict):
    ct = raw.get("coolant_temp")
    ot = raw.get("oil_temp")
    op = raw.get("oil_pressure")
    maf = raw.get("maf")
    mapk = raw.get("map_kpa")
    bv = raw.get("battery_v")
    vib = raw.get("vibratii_relanti")
    zile = raw.get("zile_de_la_ultima_revizie")
    age = raw.get("vechime_ani")

    if ct is not None and ot is not None:
        raw["delta_temp"] = float(ot - ct)

    if zile is not None:
        raw["service_overdue"] = int(float(zile) >= 180)

    if op is not None:
        raw["pressure_low"] = int(float(op) < 1.20)
        raw["pressure_very_low"] = int(float(op) < 0.90)

    if ct is not None:
        raw["coolant_overheat"] = int(float(ct) > 102.0)
    if ot is not None:
        raw["oil_overheat"] = int(float(ot) > 115.0)

    if bv is not None:
        raw["battery_low"] = int(float(bv) < 12.2)
    if vib is not None:
        raw["vibratii_high"] = int(float(vib) > 3.2)

    if maf is not None and mapk is not None:
        maf_expected = 2.6 + 0.20 * float(mapk)
        raw["airflow_mismatch"] = int((float(maf) < (maf_expected - 2.2)) and (float(mapk) > 55.0))

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

    keys = [
        "coolant_overheat", "oil_overheat",
        "pressure_low", "pressure_very_low",
        "battery_low", "vibratii_high",
        "airflow_mismatch", "service_overdue"
    ]
    if all(k in raw for k in keys):
        raw["anomaly_count"] = int(sum(int(raw[k]) for k in keys))

    if "delta_temp" in raw and ct is not None and ot is not None:
        dt = float(raw["delta_temp"])
        raw["temp_stress"] = float(
            max(0.0, float(ct) - 98.0) / 20.0
            + max(0.0, float(ot) - 110.0) / 25.0
            + max(0.0, dt - 18.0) / 18.0
        )

    if op is not None and "pressure_residual" in raw:
        pr = float(raw["pressure_residual"])
        raw["pressure_stress"] = float(
            max(0.0, 1.30 - float(op)) / 1.30
            + max(0.0, -pr) / 1.20
        )

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

    ct = raw.get("coolant_temp")
    ot = raw.get("oil_temp")
    op = raw.get("oil_pressure")
    bv = raw.get("battery_v")
    vib = raw.get("vibratii_relanti")
    zile = raw.get("zile_de_la_ultima_revizie")
    maf = raw.get("maf")
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


def build_raw_from_ui(ui: dict):
    return {
        "km": float(ui["km"]),
        "vechime_ani": float(ui["vechime_ani"]),
        "zile_de_la_ultima_revizie": float(ui["zile_revizie"]),
        "map_kpa": float(ui["map_kpa"]),
        "maf": float(ui["maf"]),
        "coolant_temp": float(ui["coolant_temp"]),
        "oil_temp": float(ui["oil_temp"]),
        "oil_pressure": float(ui["oil_pressure"]),
        "battery_v": float(ui["battery_v"]),
        "vibratii_relanti": float(ui["vibratii_relanti"]),
    }


st.set_page_config(page_title="Predictie defect auto (RN) - V5", page_icon="🚗", layout="centered")
st.title("🚗 Predictie defect auto (RN)")
st.caption("Model demonstrativ pe date simulate. Rezultatul este orientativ.")

root = get_root()

try:
    feature_names, mean, scale, scaler_path = load_scaler(root)
    model, meta, model_path = load_model(root)
    final_metrics, final_metrics_path = load_final_metrics(root)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.expander("Detalii model (informativ)", expanded=False):
    st.write(f"Model: `{model_path}`")
    st.write(f"Scaler: `{scaler_path}`")
    st.write(f"Features: **{len(feature_names)}**")
    st.write(f"Prag: **{meta['thr']:.3f}** ({meta['thr_src']})")
    st.write(f"Best AUC (val): **{meta['best_auc']:.4f}** | Best epoch: **{meta['best_epoch']}**")
    st.write(f"Loss: **{meta['loss_name']}** | max_fpr: **{meta['max_fpr']:.3f}**")

    if final_metrics is not None:
        acc = final_metrics.get("accuracy", None)
        f1 = final_metrics.get("f1", None)
        auc = final_metrics.get("auc", None)
        if acc is not None or f1 is not None or auc is not None:
            st.write("Metrici pe TEST (din results/final_metrics.json):")
            if acc is not None:
                st.write(f"- Accuracy: **{float(acc):.4f}**")
            if f1 is not None:
                st.write(f"- F1: **{float(f1):.4f}**")
            if auc is not None:
                st.write(f"- AUC: **{float(auc):.4f}**")
    else:
        st.write(f"Metrici pe TEST: lipseste `{final_metrics_path}` (ruleaza evaluare_model.py).")

st.subheader("Masurare curenta")

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        km = st.number_input("Kilometraj total (km)", min_value=0, value=150000, step=1000)
        vechime_ani = st.number_input("Vechime (ani)", min_value=0, value=8, step=1)
        zile_revizie = st.number_input("Zile de la ultima revizie", min_value=0, value=120, step=10)
        map_kpa = st.number_input("MAP (kPa)", min_value=0.0, value=45.0, step=1.0, format="%.2f")
        maf = st.number_input("MAF (g/s)", min_value=0.0, value=11.0, step=0.1, format="%.2f")

    with c2:
        coolant_temp = st.number_input("Temp. lichid racire (C)", min_value=0.0, value=92.0, step=0.5, format="%.2f")
        oil_temp = st.number_input("Temp. ulei (C)", min_value=0.0, value=98.0, step=0.5, format="%.2f")
        oil_pressure = st.number_input("Presiune ulei (bar)", min_value=0.0, value=2.4, step=0.05, format="%.3f")
        battery_v = st.number_input("Baterie (V)", min_value=0.0, value=13.2, step=0.05, format="%.2f")
        vibratii_relanti = st.number_input("Vibratii relanti", min_value=0.0, value=1.2, step=0.05, format="%.2f")

    submitted = st.form_submit_button("Ruleaza predictia")

if submitted:
    ui_vals = {
        "km": km,
        "vechime_ani": vechime_ani,
        "zile_revizie": zile_revizie,
        "map_kpa": map_kpa,
        "maf": maf,
        "coolant_temp": coolant_temp,
        "oil_temp": oil_temp,
        "oil_pressure": oil_pressure,
        "battery_v": battery_v,
        "vibratii_relanti": vibratii_relanti,
    }

    raw = build_raw_from_ui(ui_vals)
    raw = compute_derived(raw)

    missing = [f for f in feature_names if f not in raw]
    for f in missing:
        raw[f] = 0.0

    X = np.array([raw[f] for f in feature_names], dtype=np.float32).reshape(1, -1)
    X_scaled = (X - mean.reshape(1, -1)) / (scale.reshape(1, -1) + 1e-12)

    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32))
        prob = float(torch.sigmoid(logits).cpu().numpy().reshape(-1)[0])

    thr = float(meta["thr"])
    pred_defect = prob >= thr

    critical, recs = rule_recommendations(raw)
    min_km, max_km, km_text = estimate_km_remaining(prob, critical)

    st.subheader("📊 Rezultat")
    st.metric("Probabilitate defect", f"{prob*100:.1f}%")
    st.write(f"**Predictie:** {'❌ DEFECT PROBABIL' if pred_defect else '✅ OK'}  (prag: {thr:.3f} / {meta['thr_src']})")
    st.write(f"**Estimare km ramasi (orientativ):** {min_km} - {max_km} km")
    st.caption(km_text)

    st.subheader("🔧 Recomandari")
    if recs:
        for r in recs:
            st.write(f"- {r}")
    else:
        st.write("Nu exista semnale clare pe reguli. Daca apar simptome, fa o diagnoza OBD si verificari de rutina.")

    with st.expander("Date folosite (debug)", expanded=False):
        st.json({k: raw.get(k, None) for k in sorted(raw.keys())})
else:
    st.info("Completeaza valorile si apasa butonul: Ruleaza predictia.")
