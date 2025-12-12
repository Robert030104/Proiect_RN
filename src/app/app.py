import streamlit as st
import numpy as np

st.title("Predictie defect motor – Etapa 4")

features = [
    "km",
    "vechime_ani",
    "temperatura_motor",
    "temperatura_ulei",
    "presiune_ulei",
    "vibratii",
    "ore_de_la_revizie",
    "km_de_la_schimb_ulei",
    "maf",
    "map"
]

inputs = []
for f in features:
    inputs.append(st.number_input(f, value=0.0))

if st.button("Ruleaza predictia"):
    x = np.array(inputs)
    st.success("Rezultat: DEFECT / NORMAL (model neantrenat)")
