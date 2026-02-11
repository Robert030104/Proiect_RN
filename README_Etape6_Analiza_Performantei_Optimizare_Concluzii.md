# README – Etapa 6: Analiza Performantei, Optimizarea si Concluzii Finale

**Disciplina:** Retele Neuronale  
**Institutie:** POLITEHNICA Bucuresti – FIIR  
**Student:** Pintea Robert-Stefan 
**Link Repository GitHub:** https://github.com/Robert030104 
**Data predarii:** 22.01.2026

---

## Scopul Etapei 6

Aceasta etapa corespunde punctelor **7. Analiza performantei si optimizarea parametrilor**, **8. Analiza si agregarea rezultatelor** si **9. Formularea concluziilor finale**.

**Obiectiv principal:** optimizarea modelului RN pentru predictia defectelor auto (OBD-like) si maturizarea aplicatiei (pipeline complet + prag decizie realist + UI).

**Pornire (Etapa 5):**
- pipeline functional: `dataset.py -> procesare_dataset.py -> split.py -> train_model.py -> evaluare_model.py`
- model PyTorch (MLP) + scaler salvat
- inferenta integrata in Streamlit (UI)
- evaluare cu raport clar (TN/FP/FN/TP + metrici explicate)

---

## PREREQUISITE – Verificare Etapa 5 (indeplinit)

- [x] Model antrenat salvat in `models/model_predictie_defecte.pth`  
- [x] Metrici baseline raportate (Accuracy ≥65%)  
- [x] Pipeline complet functional (raw -> processed -> split -> train -> eval)  
- [x] UI functional (Streamlit) care incarca modelul si face inferenta  
- [x] State Machine / regula de decizie implementata prin prag calibrat cu constraint pe FPR (max alarme false)

---

## Rezumat rapid – Ce am optimizat in Etapa 6

In Etapa 6 am trecut de la un model “merge pe date usoare” la un model **realist**, unde clasele OK/Defect se suprapun partial, iar decizia este calibrata industrial prin constrangere pe alarme false.

### Optimizari cheie:
1. **Dataset V5 imbunatatit** (simulare mai realista, features derivate + trend-uri)  
2. **Corectarea pipeline-ului de split** (rezolvare mismatch coloane intre processed si X_val/X_test)  
3. **Antrenare + calibrare prag automata** folosind regula `max_fpr` (ex: 0.13 ≈ 12-13 alarme false / 100 OK)  
4. **Tuning parametri model**: pos_mult (class weighting), max_fpr, loss (BCE vs Focal), early stopping, scheduler fix  
5. **UI actualizata**: calcul automat features derivate (fara bifat manual), afisare probabilitate + recomandari + estimare km

---

# 1. Actualizarea Aplicatiei Software in Etapa 6

## 1.1 Tabel Modificari Aplicatie Software

| Componenta | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|---|---|---|---|
| Model incarcat | `data/model_rn.pt` (vechi path) | `models/model_predictie_defecte.pth` (checkpoint PyTorch) | standardizare path + eliminare erori “FileNotFoundError” |
| Features in UI | unele bifate manual / incomplete | calcul automat: `pressure_low`, `service_overdue`, `temp_stress`, `trend_*`, `health_index` | user nu trebuie sa “ghiceasca” derived features |
| Prag decizie | prag fix (ex 0.60) | prag **calibrat automat** pe validation cu regula `max_fpr` | control real alarme false |
| State machine | OK vs DEFECT simplu | prag + recomandari rule-based pentru semnale critice | explicabilitate + interpretare pentru utilizator |
| Evaluare | output dificil | raport clar: TN/FP/FN/TP + Accuracy/Precision/Recall/FPR | interpretabil pentru proiect |
| Pipeline | mismatch coloane la split (22 vs 33) | split corect pe `dataset_processed.csv` complet | rezolva “Lipsesc coloane” la train |

## 1.2 Pipeline end-to-end re-testat

Test complet rulat local:
- `py src/data_acquisition/dataset.py` -> salveaza `data/raw/dataset_auto.csv`
- `py src/preprocesing/procesare_dataset.py` -> salveaza `data/processed/dataset_processed.csv` + `scaler.pkl`
- `py src/preprocesing/split.py` -> salveaza `data/train|validation|test/X_*.csv, y_*.csv`
- `py src/neural_network/train_model.py` -> salveaza `models/model_predictie_defecte.pth`
- `py src/neural_network/evaluare_model.py` -> raport clar pe test
- `streamlit run src/app/app.py` -> inferenta interactiva

---

# 2. Analiza Detaliata a Performantei

## 2.1 Confusion Matrix (model final ales Etapa 6)

**Configuratie finala folosita pentru decizie industriala:**  
- `loss = BCE`
- `pos_mult = 1.3`
- `max_fpr = 0.13` (tinta ~12-13 alarme false / 100 OK)
- prag rezultat automat: **0.563**

**Evaluare TEST (clar):**
- Test samples: 1800, defect rate: 25.28%
- ROC-AUC: **0.7082**
- PR-AUC: **0.5597**
- Accuracy: **0.771**
- Precision (Defect): **0.558**
- Recall (Defect): **0.457**
- FPR: **0.123** (≈ 12.3 alarme false la 100 OK)

**Confusion Matrix (TN/FP/FN/TP):**
- TN = 1180
- FP = 165  (alarme false)
- FN = 247  (defecte ratate)
- TP = 208  (defecte prinse)

### Interpretare

**Clasa cu performanta mai buna:** OK  
- TNR ≈ 0.877 (majoritatea OK sunt lasate in pace)

**Clasa mai dificila:** Defect  
- Recall ≈ 0.457 (dataset realist -> defectele “soft” se suprapun cu OK)

**Confuzia principala:** Defect real prezis OK (FN)  
- Cauza: overlap intre semnale (defecte incipiente, valori inca “aproape normale”)

**Impact industrial (proiect):**
- preferam un prag calibrat pe FPR (alarme false controlate), acceptand ca defectele foarte incipiente pot fi ratate
- pentru a prinde mai multe defecte se poate creste `max_fpr` (trade-off)

---

## 2.2 Analiza 5 exemple gresite (cauzal)

> Nota: exemplele sunt reprezentative tipologic (FN/FP). In proiect, analiza se bazeaza pe inspectia features-urilor din randurile respective.

### Exemplu 1 (FN): defect real, prezis OK (probabilitate aproape de prag)
**Cauza probabila:** defect incipient, fara semnal clar pe presiune / temp.  
**Indicii:** `oil_pressure_bar` > 1.2, `temp_stress` mic, `anomaly_count` mic.  
**Solutie:** crestere sensibilitate (max_fpr 0.14–0.16) sau crearea de features mai predictive (trend-uri mai puternice).

### Exemplu 2 (FN): defect real, trend-uri lipsa / zero
**Cauza probabila:** fara masurarea anterioara, `trend_* = 0`, model pierde semnal.  
**Solutie:** UI sa incurajeze introducerea masurarii anterioare (t0) pentru V5.

### Exemplu 3 (FP): OK prezis defect (alarma falsa)
**Cauza probabila:** `pressure_residual` negativ mic + `vibratii_relanti` crescute (semnale “slabe” dar combinate).  
**Solutie:** prag mai strict (max_fpr 0.12) sau introducere stare “WATCHLIST” (monitorizare, nu alarma).

### Exemplu 4 (FP): MAP mare + MAF relativ mic (airflow_mismatch)
**Cauza:** model interpreteaza risc de admisie/boost leak, dar poate fi variatie normala.  
**Solutie:** rafinare formula `airflow_mismatch` sau reducere pondere in dataset pentru acest semnal.

### Exemplu 5 (FN): presiune ulei borderline (1.20–1.30)
**Cauza:** zona de tranzitie (borderline) are overlap mare; pragul pe FPR impune conservatorism.  
**Solutie:** state machine cu “WARN” pentru borderline: daca probabilitate 0.45–0.55 -> recomandare de verificare, nu verdict dur.

---

# 3. Optimizarea Parametrilor si Experimentare

## 3.1 Strategia de optimizare adoptata

**Abordare:** manual + iterativ (experimentare sistematica, modificare 1-2 parametri odata)  
**Criteriu:** model stabil + prag calibrat pe FPR (alarme false controlate), nu doar accuracy.

**Axe explorate:**
1. Dataset realism (features derivate + trend/stress)  
2. pos_mult (class weighting)  
3. max_fpr (calibrare prag)  
4. loss function (BCE vs focal)  
5. training stability (scheduler/early stop)

---

## 3.2 Tabel Experimente (minimum 4)

| Exp# | Modificare fata de Baseline | Rezultat cheie | Observatii |
|---|---|---|---|
| Baseline (E5) | Dataset mai usor (separare mai buna) | ROC-AUC ~0.80 | foarte bun pe simulare mai “curata”, dar mai putin realist |
| Exp 1 | Dataset V5 realist + features derivate/trend | AUC a scazut ~0.68–0.71 | normal: overlap OK/Defect mai mare |
| Exp 2 | Fix pipeline (split) + trend features | train fara erori “Lipsesc coloane” | X_val avea 22 cols, processed 33 -> rezolvat |
| Exp 3 | pos_mult=1.2, max_fpr=0.115 | Recall ~0.43, FPR ~0.11 | prag strict, mai putine alarme false |
| Exp 4 | pos_mult=1.35, max_fpr=0.115 | aproape acelasi recall, AUC usor mai mic | cresterea pos_mult nu a adus castig semnificativ |
| Exp 5 (Final) | pos_mult=1.3 + max_fpr=0.13 | Accuracy 0.771, Recall 0.457, FPR 0.123 | trade-off bun (≈12.3 alarme false / 100 OK) |

**Justificare alegere finala (Exp 5):**
1. Pastreaza control industrial asupra alarmelor false (FPR ≈ 0.123)  
2. Creste recall fata de setari mai stricte (prinde mai multe defecte)  
3. Rezultatul e stabil si interpretabil (raport clar + prag automat)

---

# 4. Agregarea Rezultatelor

## 4.1 Rezultate finale (Etapa 6)

| Metica | Etapa 5 (dataset mai usor) | Etapa 6 (V5 realist) | Status |
|---|---:|---:|---|
| Accuracy | ~0.78 | **0.771** | OK (≥0.70) |
| ROC-AUC | ~0.80 | **0.708** | realist / moderat |
| PR-AUC | ~0.59–0.65 | **0.560** | OK pentru defect rate 25% |
| Precision (Defect) | ~0.57–0.67 | **0.558** | OK |
| Recall (Defect) | ~0.43–0.54 (in functie de max_fpr) | **0.457** | moderat |
| Alarme false / 100 OK | 11–13 | **12.3** | conform setarii max_fpr |

---

# 5. Concluzii Finale si Lectii Invatare

## 5.1 Evaluare sintetica

**Obiective atinse:**
- [x] Model RN functional cu Accuracy > 70% (0.771)  
- [x] Integrare completa in aplicatie (dataset->preprocess->train->eval->UI)  
- [x] Prag decizie calibrat automat pe `max_fpr` (control alarme false)  
- [x] Pipeline end-to-end testat local  
- [x] UI afiseaza probabilitate defect + recomandari + estimare km

**Obiective partial atinse:**
- [ ] Recall defect > 0.60 la FPR ~0.12 (necesita fie acceptarea mai multor alarme false, fie date mai separabile)

---

## 5.2 Limitari identificate

1. **Date simulate (nu reale):** chiar daca sunt realiste, nu reflecta perfect toate cazurile OBD.  
2. **Overlap intre OK/Defect:** defectele incipiente au semnale apropiate de normal -> limita AUC/recall.  
3. **Trend-uri necesita masurare anterioara:** fara t0, performanta poate scadea.  
4. **Decizia e un compromis:** cresterea recall necesita acceptarea mai multor alarme false (control prin max_fpr).

---

## 5.3 Directii viitoare

**Pe termen scurt:**
1. Imbunatatire dataset: intarirea relatiei cauzale intre defect si semnale (mai putin overlap, reducere label_noise).  
2. Introducere stare “WATCHLIST” (ex: probabilitate 0.45–0.55) pentru borderline.  
3. Export ONNX (optional) pentru latenta mai mica.

**Pe termen mediu:**
1. Date reale OBD (loguri) pentru finetuning  
2. Drift monitoring (daca se schimba distributia senzorilor)  
3. Calibrare probabilitati (Platt/Isotonic) pentru interpretare mai buna a procentului.

---

## 5.4 Lectii invatate

**Tehnice:**
1. Calitatea datasetului si feature engineering (trend/stress) au impact mai mare decat cresterea arhitecturii MLP.  
2. Pragul optim nu trebuie fix: calibrat pe `max_fpr` (constrangere industriala) e mai realist.  
3. Epoci multe nu cresc AUC daca datele nu separa clasele; early stopping e esential.

**Proces:**
1. Debug pipeline (paths, split, coloane) e critic: un model bun cade daca pipeline-ul are mismatch.  
2. Evaluarea “clara” (TN/FP/FN/TP) e esentiala ca sa intelegi trade-off-ul.  
3. Iteratiile scurte (o modificare o data) au dus la progres controlat.

---

## 5.5 Plan post-feedback 

Dupa feedback:
1. Daca se cere recall mai mare: cresc `max_fpr` la 0.14–0.16 sau introduc “WARN/WATCHLIST”.  
2. Daca se cere imbunatatire date: ajustez `dataset.py` (reduc overlap, cresc impactul trend_pressure/pressure_stress).  
3. Daca se cere claritate UI: adaug vizualizare bar/progress + explicatii pentru recomandari.  
4. Actualizez README Etapa 3–5 pentru a reflecta versiunea finala V5.

**Commit final:** `"Versiune finala examen - toate corectiile implementate"`  
**Tag final:** `v1.0-final-exam`

---

## Structura repository la finalul Etapei 6 

Proiect_RN/
├── README.md
├── etapa6_optimizare_concluzii.md
├── data/
│ ├── raw/dataset_auto.csv
│ ├── processed/dataset_processed.csv
│ ├── train/X_train.csv y_train.csv
│ ├── validation/X_validation.csv y_validation.csv
│ └── test/X_test.csv y_test.csv
├── models/
│ └── model_predictie_defecte.pth
├── scaler.pkl
├── src/
│ ├── data_acquisition/dataset.py
│ ├── preprocesing/procesare_dataset.py
│ ├── preprocesing/split.py
│ ├── neural_network/model.py
│ ├── neural_network/train_model.py
│ ├── neural_network/evaluare_model.py
│ └── app/app.py
└── requirements.txt


---

## Instructiuni de rulare 

```bash
# 1) Genereaza date
py src/data_acquisition/dataset.py

# 2) Preprocesare + scaler
py src/preprocesing/procesare_dataset.py

# 3) Split
py src/preprocesing/split.py

# 4) Train
py src/neural_network/train_model.py

# 5) Evaluare
py src/neural_network/evaluare_model.py

# 6) UI
streamlit run src/app/app.py
