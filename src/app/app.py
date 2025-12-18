import sys
from pathlib import Path

# =========================
# FIX PYTHON PATH
# =========================
ROOT = Path(__file__).resolve().parents[1]  # src/
sys.path.append(str(ROOT))

# =========================
# IMPORTURI
# =========================
import streamlit as st
import numpy as np
import torch
import joblib

from neural_network.model import DefectPredictor

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Predictie defect auto",
    layout="centered"
)

st.title("🔧 Sistem inteligent de predictie defect auto")
st.markdown("Model bazat pe Rețele Neuronale (RN) – mentenanță predictivă")

# =========================
# ÎNCĂRCARE MODEL + SCALER
# =========================
@st.cache_resource
def load_model_and_scaler():
    scaler = joblib.load("config/scaler.pkl")
    input_dim = scaler.n_features_in_

    model = DefectPredictor(input_dim)
    model.load_state_dict(
        torch.load("models/model_predictie_defecte.pth", map_location="cpu")
    )
    model.eval()

    return model, scaler


model, scaler = load_model_and_scaler()

# =========================
# FORMULAR INPUT
# =========================
st.subheader("📥 Date de intrare vehicul")

km = st.number_input("Kilometraj total (km)", 0, 500_000, 120_000)
vechime = st.number_input("Vechime vehicul (ani)", 0, 30, 8)
temp_motor = st.number_input("Temperatura motor (°C)", 50.0, 150.0, 90.0)
temp_ulei = st.number_input("Temperatura ulei (°C)", 50.0, 160.0, 95.0)
presiune_ulei = st.number_input("Presiune ulei (psi)", 0.0, 100.0, 45.0)
vibratii = st.number_input("Vibrații motor (mm/s)", 0.0, 5.0, 0.6)
ore_revizie = st.number_input("Ore de la ultima revizie", 0, 3000, 400)
km_schimb_ulei = st.number_input("Km de la ultimul schimb de ulei", 0, 50_000, 8_000)
maf = st.number_input("Debit aer MAF (g/s)", 0.0, 10.0, 2.5)
map_val = st.number_input("Presiune MAP (bar)", 0.0, 3.0, 1.2)

# =========================
# PREDICȚIE
# =========================
if st.button("🚀 Rulează predicția"):
    X = np.array([[
        km,
        vechime,
        temp_motor,
        temp_ulei,
        presiune_ulei,
        vibratii,
        ore_revizie,
        km_schimb_ulei,
        maf,
        map_val
    ]])

    # scalare
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # inferență
    # inferență
with torch.no_grad():
    output = model(X_tensor)

    # pentru model cu 2 clase
    probs = torch.softmax(output, dim=1)
    prob_defect = probs[0, 1].item()


    # verdict
    st.markdown("---")
    st.subheader("📊 Rezultat")

    if prob_defect >= 0.5:
        st.error(f"❌ DEFECT PROBABIL")
        st.metric("Probabilitate defect", f"{prob_defect*100:.2f}%")

        estimare_km = int((1 - prob_defect) * 10_000)
        st.warning(
            f"🔔 Recomandare: efectuați revizia în aproximativ **{estimare_km} km**."
        )
    else:
        st.success("✅ FUNCȚIONARE NORMALĂ")
        st.metric("Probabilitate defect", f"{prob_defect*100:.2f}%")

        estimare_km = int((1 - prob_defect) * 25_000)
        st.info(
            f"✔ Estimare până la următoarea revizie: **{estimare_km} km**."
        )

    st.caption(
        "⚠️ Rezultatul este orientativ. Modelul este antrenat pe date simulate."
    )
