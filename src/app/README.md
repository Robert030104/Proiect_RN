# 📘 README – App

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pintea Robert Stefan  
**Link Repository GitHub:** https://github.com/Robert030104/Proiect_RN  
**Data:** 12/02/2026  

## Descriere

Acest modul reprezinta interfata utilizator (UI) a aplicatiei de predictie defect auto,
construita cu Streamlit.

Aplicatia permite:
- Introducerea manuala a valorilor senzorilor (simulare OBD-like)
- Calculul probabilitatii de defect folosind modelul RN optimizat
- Afisarea deciziei (OK / DEFECT)
- Afisarea pragului utilizat (threshold_f1 / threshold_auto)
- Generarea de recomandari tehnice pe baza regulilor definite
- Estimare orientativa a kilometrilor ramasi pana la interventie

Modelul utilizat este `models/optimized_model.pt`
Scaler-ul utilizat este `config/scaler.pkl`

---

## Arhitectura UI

Fluxul aplicatiei:

1. Input date utilizator
2. Construire vector features
3. Standardizare cu scaler salvat
4. Forward pass prin MLP
5. Aplicare threshold
6. Afisare rezultat + recomandari

---

## Cerinte

Python >= 3.10

Dependente (vezi requirements.txt):
- torch
- streamlit
- numpy
- pandas
- scikit-learn

---

## Instalare

Din root-ul proiectului:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
