import sys
from pathlib import Path

import streamlit as st
import numpy as np
import torch
import joblib

def find_project_root(start: Path) -> Path:
    p = start
    for _ in range(8):
        if (p / "config").exists() and (p / "models").exists():
            return p
        p = p.parent
    return start

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(APP_DIR)

SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
else:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_network.model import DefectPredictor  # noqa: E402


st.set_page_config(page_title="Predictie defect auto", layout="centered")

st.title("Sistem inteligent de mentenanta predictiva")
st.markdown(
    "Aplicatie bazata pe Retele Neuronale pentru estimarea riscului de defect "
    "si recomandarea momentului optim pentru revizie."
)

SCALER_PATH = PROJECT_ROOT / "config" / "scaler.pkl"
MODEL_PATH = PROJECT_ROOT / "models" / "model_predictie_defecte.pth"


@st.cache_resource
def load_model_and_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler lipsa: {SCALER_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model lipsa: {MODEL_PATH}")

    scaler = joblib.load(SCALER_PATH)
    input_dim = int(getattr(scaler, "n_features_in_", 0))
    if input_dim <= 0:
        raise ValueError("Scaler invalid (n_features_in_ lipsa sau <= 0).")

    model = DefectPredictor(input_dim)
    state = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    return model, scaler


try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error("Nu am putut incarca modelul/scaler-ul.")
    st.code(str(e))
    st.stop()


st.subheader("Date de intrare vehicul")

km = st.number_input("Kilometraj total (km)", min_value=0, max_value=500_000, value=120_000, step=1000)
vechime = st.number_input("Vechime vehicul (ani)", min_value=0, max_value=30, value=8, step=1)
temp_motor = st.number_input("Temperatura motor (C)", min_value=50.0, max_value=150.0, value=90.0, step=0.5)
temp_ulei = st.number_input("Temperatura ulei (C)", min_value=50.0, max_value=160.0, value=95.0, step=0.5)
presiune_ulei = st.number_input("Presiune ulei (psi)", min_value=0.0, max_value=100.0, value=45.0, step=0.5)
vibratii = st.number_input("Vibratii motor (mm/s)", min_value=0.0, max_value=5.0, value=0.6, step=0.05)
ore_revizie = st.number_input("Ore de la ultima revizie", min_value=0, max_value=3000, value=400, step=10)
km_schimb_ulei = st.number_input("Km de la ultimul schimb de ulei", min_value=0, max_value=50_000, value=8_000, step=500)
maf = st.number_input("Debit aer MAF (g/s)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
map_val = st.number_input("Presiune MAP (bar)", min_value=0.0, max_value=3.0, value=1.2, step=0.05)

if st.button("Ruleaza predictia"):
    X = np.array(
        [[
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
        ]],
        dtype=np.float64
    )

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error("Scaler.transform a esuat. Verifica ordinea/numarul de feature-uri.")
        st.code(str(e))
        st.stop()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)
        probs = torch.softmax(output, dim=1)

        if probs.shape[1] < 2:
            st.error(f"Modelul nu pare sa fie pe 2 clase (shape={tuple(probs.shape)}).")
            st.stop()

        prob_defect = float(probs[0, 1].item())

    st.markdown("---")
    st.subheader("Rezultat analiza")

    st.metric("Probabilitate defect", f"{prob_defect * 100:.2f}%")

    if prob_defect >= 0.5:
        st.error("DEFECT PROBABIL")
        estimare_km = int((1.0 - prob_defect) * 10_000)
        st.warning(f"Recomandare: efectuati revizia in aproximativ {estimare_km} km.")
    else:
        st.success("FUNCTIONARE NORMALA")
        estimare_km = int((1.0 - prob_defect) * 25_000)
        st.info(f"Estimare pana la urmatoarea revizie: {estimare_km} km.")

    st.progress(prob_defect)

    st.caption(
        "Rezultatul este orientativ. Modelul este antrenat pe date simulate "
        "si are scop demonstrativ/educational."
    )
