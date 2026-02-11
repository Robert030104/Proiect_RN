# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pintea Robert Stefan  
**Link Repository GitHub:** https://github.com/Robert030104/Proiect_RN  
**Data:** 12/12/2025  

---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe Rețele Neuronale**
din specificațiile proiectului.

Scopul este realizarea unui **schelet complet, coerent și funcțional** al unui Sistem cu Inteligență Artificială (SIA),
care demonstrează rularea pipeline-ului complet:
**achiziție date → preprocesare → inferență RN → afișare rezultat**.

În această etapă:
- modelul de rețea neuronală este **doar definit și compilat**
- NU se urmărește performanță ridicată
- inferența este demonstrativă (model neantrenat sau dummy)

---

## 1. Nevoie Reală → Soluție SIA → Modul Software

| Nevoie reală concretă | Cum o rezolvă SIA-ul | Modul software responsabil |
|----------------------|----------------------|----------------------------|
| Detectarea timpurie a defectelor la motoare auto | Analiza parametrilor de funcționare (temperaturi, presiuni, vibrații, airflow) și clasificare normal / defect | Data Acquisition + RN |
| Reducerea costurilor de mentenanță neplanificată | Predicție stării tehnice pe baza datelor istorice și curente | RN |
| Suport decizional pentru utilizator | Afișarea probabilității de defect într-o interfață simplă | UI / Web Service |

---

## 2. Contribuția Originală la Setul de Date

### Contribuția originală

- **Total observații (Etapa 4):** 2000  
- **Observații originale:** 2000 (100%)

### Tipul contribuției

- [x] Date generate prin simulare software realistă  
- [ ] Date achiziționate cu senzori proprii  
- [ ] Etichetare manuală  
- [ ] Date externe preluate din alte surse  

### Descriere detaliată

Setul de date a fost generat integral prin cod Python, utilizând distribuții statistice
și corelații inspirate din funcționarea reală a motoarelor auto.

Parametrii simulați includ:
- kilometraj și vechime
- temperaturi motor și ulei
- presiune ulei
- vibrații
- senzori MAF și MAP
- parametri de mentenanță (ore / km de la revizie)

Eticheta **`defect`** este determinată printr-un model logic care combină mai mulți factori
critici (ex: temperatură ridicată + presiune ulei scăzută + vibrații crescute).

### Locații relevante

- **Cod generare date:** `src/data_acquisition/dataset.py`  
- **Dataset generat:** `data/raw/dataset_auto.csv`

---

## 3. Diagrama State Machine a Sistemului

### Justificarea State Machine-ului

A fost aleasă o arhitectură de tip **monitorizare și predicție batch**, adecvată aplicațiilor
de mentenanță predictivă, unde datele sunt:
- introduse manual de utilizator
- sau citite din fișiere CSV / batch-uri periodice

Această abordare permite control clar al fluxului și integrarea facilă a modelului RN.

### Stările principale

1. **IDLE**  
   Sistemul așteaptă date de intrare.

2. **ACQUIRE_DATA**  
   Datele sunt citite din fișier CSV sau introduse manual prin UI.

3. **PREPROCESS**  
   Datele sunt validate și scalate folosind parametrii salvați (scaler).

4. **RN_INFERENCE**  
   Modelul de rețea neuronală realizează predicția (în Etapa 4 – model neantrenat).

5. **DISPLAY_RESULT**  
   Rezultatul (normal / defect) este afișat utilizatorului.

6. **ERROR**  
   Gestionarea erorilor (date invalide, valori în afara domeniului).

Starea **ERROR** este esențială pentru robustețea sistemului și evitarea opririlor necontrolate.

Diagrama State Machine este salvată în:
`docs/state_machine.png`

---

## 4. Arhitectura Modulară a Sistemului SIA

### Modul 1: Data Logging / Data Acquisition

- Generează date originale prin simulare
- Exportă datele în format CSV
- Structura dataset-ului este compatibilă cu pipeline-ul de preprocesare
- Codul rulează fără erori

**Locație:** `src/data_acquisition/`

---

### Modul 2: Neural Network Module

- Arhitectura rețelei neuronale este definită și compilată
- Modelul este salvat într-un fișier `.h5`
- Greutățile sunt inițializate aleator (model neantrenat)
- Modulul este pregătit pentru antrenare în Etapa 5

**Locație:** `src/neural_network/`  
**Model:** `models/untrained_model.h5`

---

### Modul 3: Web Service / UI

- Interfață simplă realizată cu **Streamlit**
- Permite introducerea manuală a parametrilor vehiculului
- Afișează rezultatul predicției (normal / defect)
- Demonstrează fluxul complet al State Machine-ului

**Locație:** `src/app/`  
**Screenshot demo:** `docs/screenshots/ui_demo.png`

---

## 5. Structura Repository-ului (final Etapa 4)

proiect-rn-pintea-robert/
├── data/
│ ├── raw/
│ ├── processed/
│ ├── generated/
│ ├── train/
│ ├── validation/
│ └── test/
├── src/
│ ├── data_acquisition/
│ ├── preprocessing/
│ ├── neural_network/
│ └── app/
├── models/
│ └── untrained_model.h5
├── config/
├── docs/
│ ├── state_machine.png
│ └── screenshots/
│ └── ui_demo.png
├── README.md
├── etapa3_analiza_date.md
├── etapa4_arhitectura_sia.md
└── requirements.txt


---

## 6. Stare Etapă

- [x] Arhitectură SIA complet definită
- [x] State Machine documentat și implementat logic
- [x] Modul Data Acquisition funcțional
- [x] Modul RN definit și compilat (neantrenat)
- [x] Modul UI funcțional cu model dummy
- [x] Pipeline complet demonstrat (date → rezultat)
- [x] Proiect pregătit pentru Etapa 5 – Antrenare Model
