# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pintea Robert Stefan  
**Link Repository GitHub:** https://github.com/Robert030104/Proiect_RN  
**Data:** 19/12/2025  

Data predării: 12.02.2026

# Scopul Etapei 5

Această etapă corespunde punctului 6. Configurarea și antrenarea modelului RN din lista de 9 etape.

# Obiectiv principal

Antrenarea efectivă a modelului RN (MLP PyTorch) pentru predicția defectelor auto pe date OBD-like simulate realist, evaluarea performanței pe setul de test și integrarea modelului antrenat în aplicația Streamlit.

# PREREQUISITE – Verificare Etapa 4 (ÎNDEPLINIT)

✔ State Machine definit și documentat
✔ Contribuție ≥40% date originale (dataset generat în dataset.py)
✔ Modul Data Logging funcțional (generează CSV raw)
✔ Modul RN definit (MLP PyTorch)
✔ UI Streamlit funcțional (inițial cu model dummy)

# Pregătire Date pentru Antrenare

# Pipeline complet rulat:

py src/data_acquisition/dataset.py
py src/preprocesing/procesare_dataset.py
py src/preprocesing/split.py

Split realizat:

70% Train (8400 samples)

15% Validation (1800 samples)

15% Test (1800 samples)

random_state = 42

stratificat pe variabila defect

Defect rate final dataset: ≈ 23%

Nivel 1 – Obligatoriu
✔ Antrenare Model

Model: MLP PyTorch

Antrenat cu:

py src/neural_network/train_model.py


Model salvat în:

models/trained_model.pt

✔ Hiperparametri Folosiți
Hiperparametru	Valoare Aleasă	Justificare
Learning rate	0.001	Valoare stabilă pentru Adam, convergență rapidă
Batch size	32	Echilibru între stabilitate gradient și timp antrenare
Epochs	50 (early stopping activ)	Permite convergență fără overfitting
Optimizer	Adam	Adaptive learning rate, potrivit pentru MLP
Loss	BCEWithLogitsLoss	Clasificare binară
Activation	ReLU (hidden) + Sigmoid (output implicit)	ReLU pentru non-linearitate, Sigmoid pentru probabilitate
Justificare Batch Size

Avem 8400 samples train → 8400 / 32 ≈ 262 batch-uri / epocă.
Batch 32 oferă:

gradient stabil

timp de antrenare rezonabil

memorie RAM sigură pe CPU

Evaluare pe Test Set

Rulat cu:

py src/neural_network/evaluare_model.py

Rezultate finale (Test Set)

Accuracy: 0.770

Precision (macro): 0.666

Recall (macro): 0.665

F1-score (macro): 0.665

ROC-AUC: 0.712

Confusion Matrix

TN = 1180

FP = 165

FN = 249

TP = 206

# Verificare Cerințe Nivel 1

✔ Accuracy ≥ 65% (0.77)
✔ F1-score macro ≥ 0.60 (0.665)
✔ Model salvat (.pt)
✔ Evaluare reală pe test
✔ UI integrat cu model antrenat

Integrare în UI (Streamlit)

Model încărcat în:

models/trained_model.pt


Scaler încărcat din:

scaler.pkl


Inferență reală:

logits = model(input_scaled)
prob = torch.sigmoid(logits)


UI afișează:

Probabilitate defect (%)

Decizie OK / DEFECT

Recomandări rule-based

Estimare km rămași

Screenshot salvat în:

docs/screenshots/inference_real.png

# Nivel 2 – Implementat

✔ Early Stopping
✔ Scheduler (ReduceLROnPlateau)
✔ Prag decizie calibrat pe validation
✔ Evaluare clară TN/FP/FN/TP
✔ Feature engineering realist (trend, stress, flags)

Indicatori Nivel 2

Accuracy = 0.77 (≥ 0.75)

F1_macro = 0.665 (sub 0.70, dar peste prag minim)

Analiză Erori (Context Industrial)
1. Clasele confundate

Majoritatea erorilor sunt:

False Negatives (FN = 249)
Defecte incipiente prezise OK.

Cauză:
Semnale borderline (presiune ulei 1.2–1.3 bar, temperaturi aproape normale).

2. Caracteristici problematice

Overlap între OK și Defect în zona presiune ulei marginală.

Trend-uri absente dacă nu există măsurare anterioară.

Defecte incipiente greu separabile liniar.

3. Impact Industrial

False Negatives → defect nedetectat → risc mecanic.

False Positives → alarmă falsă → cost verificare suplimentar.

Strategie adoptată:
Control FPR prin calibrare prag pe validation.

4. Măsuri Corective Propuse

Introducere stare „WATCHLIST” pentru probabilitate 0.45–0.55.

Creșterea ponderii trend_pressure în dataset.

Reducerea label noise pentru defecte incipiente.

Posibilă creștere max_fpr pentru recall mai mare.

# Structura Finală Repository
Proiect_RN/
├── data/
├── models/
│   └── trained_model.pt
├── scaler.pkl
├── src/
│   ├── data_acquisition/
│   ├── preprocesing/
│   ├── neural_network/
│   │   ├── model.py
│   │   ├── train_model.py
│   │   └── evaluare_model.py
│   └── app/
│       └── app.py
├── docs/
│   └── screenshots/inference_real.png
└── requirements.txt

# Checklist Final – Etapa 5

✔ Model antrenat de la zero
✔ ≥10 epoci
✔ Split 70/15/15 stratificat
✔ Accuracy ≥65%
✔ F1 macro ≥0.60
✔ UI integrat cu model real
✔ Confusion Matrix analizată
✔ Hiperparametri justificați

# Concluzie Etapa 5

Modelul RN (MLP) a fost antrenat și evaluat cu succes pe un dataset realist OBD-like.

Performanță obținută:

Accuracy: 77%

F1 macro: 0.665

ROC-AUC: 0.712

Sistemul complet (Data → Preprocess → Train → Eval → UI) este funcțional și integrat.