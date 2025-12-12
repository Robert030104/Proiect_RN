# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pintea Robert Stefan  
**Link Repository GitHub:** [link GitHub]  
**Data:** [12/12/2025]  

---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN**.
Scopul este realizarea unui **schelet complet și funcțional** al unui Sistem cu Inteligență Artificială (SIA),
care demonstrează rularea pipeline-ului complet de la date până la afișarea rezultatului.

În această etapă, modelul de rețea neuronală este **doar definit și compilat**, fără antrenare serioasă.

---

## 1. Nevoie Reală → Soluție SIA → Modul Software

| Nevoie reală concretă | Cum o rezolvă SIA-ul | Modul software responsabil |
|----------------------|----------------------|----------------------------|
| Detectarea timpurie a defectiunilor la motoare auto | Analiza parametrilor de functionare (temperaturi, presiuni, vibratii) si clasificare defect / normal | Data Acquisition + RN |
| Planificarea mentenantei preventive pentru flote auto | Predictie stare tehnica pe baza datelor istorice pentru reducerea avariilor neplanificate | RN + Web UI |

---

## 2. Contribuția Originală la Setul de Date

### Contribuția originală la setul de date:

**Total observații finale:** 2000  
**Observații originale:** 2000 (100%)

**Tipul contribuției:**
[X] Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:**

Setul de date a fost generat integral prin simulare software, utilizând distribuții statistice
realiste pentru parametri specifici funcționării motoarelor auto (kilometraj, temperaturi,
presiune ulei, vibrații, senzori MAF și MAP). Eticheta de defect a fost determinată pe baza
unor reguli logice inspirate din mentenanța reală a vehiculelor.


**Locația codului:** `src/data_acquisition/dataset.py`  
**Locația datelor:** `data/raw/dataset_auto.csv`

---

## 3. Diagrama State Machine a Sistemului

### Justificarea State Machine-ului ales:

A fost aleasă o arhitectură de tip **monitorizare și predicție batch**, deoarece sistemul
vizează evaluarea stării tehnice a unui vehicul pe baza unui set de parametri măsurați
periodic sau introduși de utilizator.

**Stările principale sunt:**
1. **IDLE** – sistemul așteaptă date de intrare  
2. **ACQUIRE_DATA** – datele sunt citite din fișier sau introduse de utilizator  
3. **PREPROCESS** – datele sunt normalizate și validate  
4. **RN_INFERENCE** – modelul de rețea neuronală realizează predicția  
5. **DISPLAY_RESULT** – rezultatul este afișat utilizatorului  
6. **ERROR** – gestionarea situațiilor de date invalide sau erori de sistem  

Starea **ERROR** este esențială deoarece datele pot fi incomplete sau în afara domeniilor
acceptate, iar sistemul trebuie să gestioneze aceste situații fără a se opri brusc.

---

## 4. Scheletul Complet al Modulelor SIA

### Modul 1: Data Logging / Acquisition

- Codul rulează fără erori
- Generează un dataset CSV complet original
- Datele sunt compatibile cu preprocesarea ulterioară
- Cod localizat în `src/data_acquisition/`

---

### Modul 2: Neural Network Module

- Arhitectura rețelei neuronale este definită și compilată
- Modelul poate fi salvat și încărcat
- Nu este necesară performanță ridicată în această etapă
- Cod localizat în `src/neural_network/`

---

### Modul 3: Web Service / UI

- Interfață simplă pentru introducerea valorilor de intrare
- Afișează un rezultat de tip defect / normal
- Implementare minimă realizată cu Streamlit
- Cod localizat în `src/app/`
- Screenshot demonstrativ salvat în `docs/screenshots/`

---

## 5. Structura Repository-ului

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
├── config/
├── docs/
│ ├── state_machine.png
│ └── screenshots/
├── README.md
├── README_Etapa3.md
├── README_Etapa4_Arhitectura_SIA.md
└── requirements.txt