# 📘 README – Neaural_network (RN)

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pintea Robert Stefan  
**Link Repository GitHub:** https://github.com/Robert030104/Proiect_RN  
**Data:** 12/02/2026  

## Descriere Generala

Acest modul implementeaza modelul de Retea Neuronala (RN) utilizat pentru
predictia defectelor auto in context de mentenanta predictiva.

Modelul este construit cu PyTorch si este antrenat de la zero
(initializare random a greutatilor).

Tip problema:
Clasificare binara (0 = normal, 1 = defect)

---

## Arhitectura Modelului (MLP)

Model utilizat: Multi-Layer Perceptron (MLP)

Structura finala:

Input (10 features numerice)
    → Linear(10 → 128)
    → BatchNorm1d(128)
    → ReLU
    → Dropout

    → Linear(128 → 64)
    → BatchNorm1d(64)
    → ReLU
    → Dropout

    → Linear(64 → 32)
    → ReLU

    → Linear(32 → 1)
Output:
    Sigmoid(logit) → Probabilitate defect

---

## Justificare Arhitectura

- Datele sunt tabulare (numerice), nu imagini sau serii temporale
- MLP este potrivit pentru relatii neliniare intre feature-uri
- BatchNorm stabilizeaza antrenarea
- Dropout reduce overfitting
- 3 straturi ascunse permit captarea interactiunilor complexe

---

## Functia de Pierdere

Focal Loss:
- alpha = 0.55
- gamma = 2.0

Motivatie:
- Setul are usoara dezechilibrare intre clase
- Focal Loss penalizeaza mai mult exemplele dificil de clasificat
- Reduce riscul de ignorare a clasei minoritare

---

## Optimizer

AdamW
- lr = 2e-4
- weight_decay = 1e-4

Motivatie:
- AdamW aplica regularizare corecta (decupled weight decay)
- Convergenta stabila
- Generalizare mai buna fata de Adam simplu

---

## Strategii de Antrenare

✔ WeightedRandomSampler (balansare clase in train)
✔ Early Stopping (patience = 45)
✔ ReduceLROnPlateau (monitor val_auc)
✔ Gradient clipping (max_norm = 2.0)

---

## Alegerea Pragului (Threshold)

Exista doua strategii:

1) threshold_auto
   - selectat pe validation set
   - respecta max_fpr = 0.13
   - potrivit pentru control alarme false

2) threshold_f1
   - selectat pentru maximizarea F1
   - utilizat in versiunea finala

Pragul este salvat in checkpoint (`optimized_model.pt`).

---

## Fisiere Principale

model.py
Defineste arhitectura MLP

train_model.py
Antrenare model
Genereaza:
- training_history.csv
- optimized_model.pt

optimize_threshold.py
Optimizeaza pragul pentru F1
Actualizeaza checkpoint-ul

evaluare_model.py
Evalueaza pe test set
Genereaza:
- final_metrics.json
- confusion_matrix_optimized.png

visualize.py
Genereaza grafice:
- loss_curve.png
- metrics_evolution.png
- accuracy_comparison.png
- f1_comparison.png


---

## Structura Iesiri

models/
    optimized_model.pt

results/
    training_history.csv
    optimization_experiments.csv
    final_metrics.json
    error_analysis.json

docs/
    results/
    optimization/

---

## Metrici Finale (Model Optimizat)

Accuracy: 98.33%
F1-Score: 0.9636
AUC: 0.9724

Verificare: results/final_metrics.json

---

## Limitari

- Dataset sintetic (nu OBD real)
- Generalizare limitata fara validare externa
- Prag dependent de obiectiv (max F1 vs control FP)

---

## Posibile Extensii

- Calibrare probabilitati (Platt scaling)
- Ensemble MLP
- Testare pe date reale OBD
- Export ONNX pentru deployment edge

---


