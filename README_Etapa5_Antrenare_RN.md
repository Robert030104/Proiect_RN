# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pintea Robert Stefan  
**Link Repository GitHub:** https://github.com/Robert030104/Proiect_RN  
**Data predării:** 12.02.2026  

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape (slide 2 din **RN Specificatii proiect.pdf**).

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței pe setul de test și integrarea modelului antrenat în aplicația completă (UI).

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

Înainte de a începe Etapa 5, a fost verificat că proiectul îndeplinește cerințele din Etapa 4:

- [x] **State Machine** definit și documentat în `docs/state_machine.png`
- [x] **Contribuție ≥40% date originale** în `data/generated/` (verificabil)
- [x] **Modul 1 (Data Logging / Data Acquisition)** funcțional - produce CSV-uri
- [x] **Modul 2 (RN)** cu arhitectură definită dar neantrenată (în Etapa 4)
- [x] **Modul 3 (UI/Web Service)** funcțional cu model dummy (Etapa 4)
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

---

## Pregătire Date pentru Antrenare

Dataset-ul folosit este cel rezultat în Etapa 3 și extins în Etapa 4/5, păstrând contribuția originală (≥40%).

**Pași de preprocesare (rezumat):**
- Split stratificat: **70% train / 15% validation / 15% test**
- `random_state = 42`
- Scalare cu **StandardScaler**, fit **doar pe train**, apoi aplicată pe validation/test (evitare data leakage)
- Scaler salvat în `config/scaler.pkl` pentru reproducibilitate și inferență în UI

**Verificare rapidă:**
```python
import pandas as pd
train = pd.read_csv("data/train/X_train.csv")
print("Train samples:", len(train))
```

---

## Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

Cerințele Nivel 1 sunt îndeplinite prin:
1. Antrenarea modelului definit în Etapa 4 pe setul final de date (≥40% originale)
2. Minimum 10 epoci (antrenare cu max_epochs=160 + early stopping)
3. Împărțire stratificată train/validation/test: 70% / 15% / 15%
4. Tabel hiperparametri + justificări (completat mai jos)
5. Metrici pe test set peste pragurile cerute (Accuracy ≥ 65%, F1 macro ≥ 0.60)
6. Salvare model antrenat în format PyTorch `.pth`
7. Integrare în UI cu inferență reală + screenshot în `docs/screenshots/inference_real.png`

---

## Tabel Hiperparametri și Justificări (OBLIGATORIU)

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------:|-----------------|
| Learning rate | 0.0007 | LR mai mic pentru stabilitate pe date tabulare și reducerea oscilațiilor |
| Batch size | 16 | Batch mic → gradient mai stabil; potrivit pentru N≈12000 |
| Number of epochs | max 160 (early stopping, patience=22) | Permite convergență; antrenarea se oprește automat dacă nu mai apare îmbunătățire |
| Optimizer | Adam | Optimizator adaptiv, potrivit pentru rețele MLP pe date tabulare |
| Weight decay | 1e-4 | Regularizare L2 pentru reducerea overfitting-ului |
| Loss function | BCEWithLogitsLoss + pos_mult=1.30 | Clasificare binară; penalizează mai mult ratarea defectelor (FN) |
| Activation functions | ReLU (hidden), Sigmoid (output) | ReLU pentru non-linearitate; Sigmoid pentru probabilitate de defect |
| Threshold rule | max_fpr=0.13 → prag=0.563 | Control al alarmelor false în context industrial (mentenanță predictivă) |

**Justificare batch size (detaliată):**
```
Am ales batch_size=16 deoarece pentru N≈12000 obținem ~750 iterații/epocă.
Batch-ul mai mic crește stabilitatea învățării și reduce zgomotul gradientului,
menținând un timp de antrenare rezonabil.
```

---

## Rezultate pe Test Set (OBLIGATORIU)

Evaluarea a fost realizată pe setul de test (clar) folosind scriptul `evaluare_model.py`.

### Rezumat metrici (test)

- **Model:** `models/model_predictie_defecte.pth`
- **Test samples:** 1800  
- **Defect rate:** 25.28%  
- **ROC-AUC:** 0.7082  
- **PR-AUC:** 0.5597  
- **Prag folosit:** regula `max_fpr=0.130` → **prag probabilitate = 0.563**

### Confusion Matrix (test)

- TN (OK prezis OK): **1180**
- FP (OK prezis Defect): **165**  *(alarme false)*
- FN (Defect prezis OK): **247**  *(defecte ratate)*
- TP (Defect prezis Defect): **208**

### Metrici principale (test)

- **Accuracy:** **0.7711**
- **F1-score (macro):** **0.6769**
- **Precision (Defect):** 0.5576
- **Recall (Defect):** 0.4571
- **FPR:** 0.123  *(≈ 12.3 alarme false la 100 vehicule OK)*

**Fișier livrabil metrici:** `results/test_metrics.json`

---

## Analiză Erori în Context Industrial (Nivel 2 – OBLIGATORIU dacă se urmărește 85–90%)

### 1. Pe ce clase greșește cel mai mult modelul?

Modelul greșește predominant pe clasa **Defect**, unde recall-ul este 0.457 (FN=247). Clasa **OK** este prezisă mai bine (TNR=0.877), deci modelul este relativ conservator: reduce alarmele false, dar poate rata defecte.

### 2. Ce caracteristici ale datelor cauzează erori?

Erorile apar în scenarii de „defect incipient”, când semnalele nu sunt extreme (presiune ulei moderat scăzută + temperaturi/vibrații în zona de tranziție). În aceste cazuri, distribuțiile claselor se suprapun, iar separarea devine dificilă, mai ales în prezența zgomotului pe vibrații sau variațiilor de sarcină (MAF/MAP).

### 3. Ce implicații are pentru aplicația industrială?

În mentenanța predictivă auto, **false negatives** sunt critice deoarece defectele ratate pot produce avarii serioase și costuri mari. **False positives** sunt mai acceptabile deoarece pot fi verificate prin diagnoză suplimentară. Modelul actual controlează FPR (0.123), dar recall pe defect este relativ mic, deci există risc de defecte ratate.

### 4. Ce măsuri corective propuneți?

1. Ajustarea pragului (ex: 0.563 → 0.45) pentru reducerea FN, acceptând o creștere moderată a FP  
2. Creșterea ponderii clasei Defect în loss (pos_mult mai mare / focal loss)  
3. Reducerea label-noise și întărirea semnalelor critice (presiune_ulei, vibrații, temp_ulei) în generator/preprocesare  
4. Feature engineering: variabile derivate (ex: `temperatura_ulei - temperatura_motor`, interacțiuni între senzori)

---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența respectă fluxul definit în State Machine-ul din Etapa 4.

| **Stare (Etapa 4)** | **Implementare (Etapa 5)** |
|---------------------|----------------------------|
| `ACQUIRE_DATA` | Citire seturi `data/train/`, `data/validation/`, `data/test/` |
| `PREPROCESS` | Aplicare `config/scaler.pkl` pe intrări (fit doar pe train) |
| `RN_INFERENCE` | Inferență cu model antrenat `models/model_predictie_defecte.pth` |
| `THRESHOLD_CHECK` | Aplicare prag 0.563 (regula max_fpr=0.13) |
| `ALERT / DISPLAY_RESULT` | Afișare rezultat în UI + probabilitate defect |

---

## Integrare în UI (OBLIGATORIU – Nivel 1)

UI (Streamlit) a fost actualizat pentru a încărca **modelul antrenat** și a face inferență reală.

**Screenshot obligatoriu:**
- `docs/screenshots/inference_real.png`

În screenshot trebuie să se vadă:
- valorile introduse
- butonul de predicție (sau acțiunea de inferență)
- rezultatul (OK/Defect) + probabilitate
- referință clară la modelul încărcat (ex: `model_predictie_defecte.pth`)

---

## Structura Repository-ului la Finalul Etapei 5

```
Proiect_RN/
├── README.md
├── docs/
│   ├── etapa5_antrenare_model.md
│   ├── state_machine.png
│   ├── loss_curve.png                  # dacă este generat (Nivel 2)
│   └── screenshots/
│       ├── ui_demo.png                 # Etapa 4
│       └── inference_real.png          # Etapa 5 (OBLIGATORIU)
├── data/
│   ├── raw/
│   ├── generated/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── neural_network/
│   └── app/
├── models/
│   └── model_predictie_defecte.pth     # model antrenat (OBLIGATORIU)
├── results/
│   ├── training_history.csv            # dacă este generat (recomandat)
│   └── test_metrics.json               # (OBLIGATORIU)
├── config/
│   └── scaler.pkl
└── requirements.txt
```

---

## Instrucțiuni de Rulare 

### 1. Setup mediu

```bash
pip install -r requirements.txt
```

### 2. Antrenare model

```bash
python train_model.py
```

> Scriptul folosește hiperparametrii: `batch_size=16`, `lr=7e-4`, `max_epochs=160`, `patience=22`, `pos_mult=1.30`, `max_fpr=0.13`.

### 3. Evaluare pe test set

```bash
python evaluare_model.py
```

### 4. Lansare UI cu model antrenat

```bash
streamlit run src/app/main.py
```

---

## Checklist Final – Înainte de Predare

### Prerequisite Etapa 4
- [x] State Machine există și e documentat
- [x] Contribuție ≥40% date originale
- [x] Cele 3 module din Etapa 4 funcționale

### Antrenare Model – Nivel 1
- [x] Model antrenat de la zero
- [x] Batch size în [8, 32] (batch_size=16)
- [x] Minimum 10 epoci (max 160 + early stopping)
- [x] Metrici test set: Accuracy ≥ 0.65, F1 macro ≥ 0.60 (obținut: 0.771 / 0.677)
- [x] Model salvat: `models/model_predictie_defecte.pth`

### Integrare UI – Nivel 1
- [ ] UI încarcă model antrenat și face inferență reală
- [ ] Screenshot: `docs/screenshots/inference_real.png`

### Fișiere rezultate
- [ ] `results/test_metrics.json` există (obligatoriu)
- [ ] `docs/loss_curve.png` (opțional – Nivel 2)
- [ ] `results/training_history.csv` (recomandat)

---

## Predare

1. Commit pe GitHub:  
   **`Etapa 5 completă – Accuracy=0.77, F1=0.68`**
2. Tag:  
   `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
3. Push:  
   `git push origin main --tags`

---

**Etapa 5 demonstrează antrenarea modelului RN, evaluarea pe setul de test și integrarea într-o aplicație SIA funcțională.**
