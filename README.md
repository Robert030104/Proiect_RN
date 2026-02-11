# 📘 README – Etapa 3: Analiza si Pregatirea Setului de Date pentru Retele Neuronale

**Disciplina:** Retele Neuronale  
**Instituție:** POLITEHNICA Bucuresti – FIIR  
**Student:** Pintea Robert Stefan  
**Data:** 20/11/2025

---

## Introducere

Acest document descrie activitatile realizate in **Etapa 3**, etapa in care se analizeaza si se preproceseaza setul de date necesar proiectului „Retele Neuronale”.
Scopul etapei este pregatirea corecta a datelor pentru instruirea modelului RN, respectand bune practici privind calitatea, consistenta si reproductibilitatea.

---

## 1. Structura Repository-ului Github (versiunea Etapei 3 – actualizata)

Proiect_RN/
├── README.md
├── docs/
│ └── datasets/ # descriere seturi de date, surse, diagrame
├── data/
│ ├── raw/ # date brute (dataset initial / generat)
│ ├── processed/ # date curatate + feature engineering
│ ├── train/ # set de instruire
│ ├── validation/ # set de validare
│ └── test/ # set de testare
├── src/
│ ├── preprocessing/ # preprocesare (curatare, scalare, split)
│ ├── data_acquisition/ # generare / achizitie date
│ └── neural_network/ # implementare RN (etapa urmatoare)
├── config/
│ └── scaler.pkl # parametri scalare salvati (fit doar pe train)
└── requirements.txt


---

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

- **Origine:** Senzori vehicule (OBD-like / semnale simulate realist)
- **Modul de achizitie:** Simulare + Generare programatica cu distributii si corelatii intre semnale
- **Perioada / conditii colectare:** Noiembrie 2025
- **Observatie importanta:** Clasa „defect” este controlata ca procent pentru a obtine un set de date utilizabil in clasificare.

### 2.2 Caracteristicile dataset-ului 

- **Numar total de observatii:** 12000
- **Numar de caracteristici (features):** 11
- **Tipuri de date:** Numerice
- **Format fisiere:** CSV

**Clasa tinta:**
- `defect` (0 = normal, 1 = defect)

**Distribuitia claselor:**
- `defect=1`: ~24–25% (controlata prin generator)
- `defect=0`: restul

### 2.3 Descrierea fiecarei caracteristici (actualizata)

| Caracteristica           | Tip     | Unitate | Descriere                                 | Domeniu valori (orientativ) |
|--------------------------|---------|---------|-------------------------------------------|-----------------------------|
| km                       | numeric | km      | Kilometraj total al vehiculului           | 10.000 – 300.000            |
| vechime_ani              | numeric | ani     | Varsta vehiculului                        | 1 – 20                      |
| temperatura_motor        | numeric | °C      | Temperatura lichid racire (coolant)       | 60 – 130                    |
| temperatura_ulei         | numeric | °C      | Temperatura ulei motor                    | 60 – 150                    |
| presiune_ulei            | numeric | psi/bar | Presiune ulei (semnal critic defect)      | 10 – 80                     |
| vibratii                 | numeric | mm/s    | Nivel vibratii motor                       | 0.1 – 3.5                   |
| ore_de_la_revizie        | numeric | ore     | Ore de la ultima revizie                   | 0 – 600                     |
| km_de_la_schimb_ulei     | numeric | km      | Km de la schimb ulei                       | 0 – 30.000                  |
| maf                      | numeric | g/s     | Debit aer masurat (MAF)                    | 5 – 400                     |
| map                      | numeric | kPa     | Presiune in galerie (MAP)                  | 20 – 120                    |
| tensiune_baterie         | numeric | V       | Tensiune baterie/alternator                | 11.0 – 14.8                 |
| defect                   | numeric | 0/1     | 1 = defect, 0 = normal                     | 0 sau 1                     |

> Daca in proiect ai si alte coloane (ex: `engine_load`, `rpm`, `throttle`, etc.), adauga-le aici identic cu numele din CSV.

**Fisier principal (raw):** `data/raw/dataset_auto.csv`  
**Fisier curatat (processed):** `data/processed/dataset_clean.csv`

---

## 3. Analiza Exploratorie a Datelor (EDA) – sintetic (actualizat)

### 3.1 Statistici descriptive aplicate

Au fost calculate statistici descriptive pentru fiecare variabila numerica:
- min / max
- medie, mediana
- deviatie standard
- quartile (Q1, Q2, Q3)

Au fost verificate distributiile (histograme) si corelatiile intre variabile (ex: temperatura_ulei vs presiune_ulei, km vs ore_de_la_revizie).

### 3.2 Analiza calitatii datelor

- Nu exista valori lipsa (dataset generat controlat) sau acestea au fost eliminate in etapa de curatare.
- Tipurile de date sunt corecte (numerice).
- Valorile se incadreaza in intervale realiste pentru parametrii unui vehicul.
- A fost verificata distributia clasei tinta si s-a confirmat un procent de defect ~25%.

### 3.3 Probleme identificate

- Exista valori extreme (outlieri) la km, maf/map, temperaturi si vibratii, ceea ce poate influenta antrenarea.
- Unele variabile au dispersie mare, deci scalarea este obligatorie.
- Unele observatii simuleaza scenarii de functionare atipice (ex: temperaturi mari + presiune mica), utile pentru invatarea clasei „defect”.

---

## 4. Preprocesarea Datelor (actualizat)

### 4.1 Curatarea datelor

- Datasetul a fost verificat de duplicate.
- Valorile lipsa au fost verificate.
- Outlierii au fost analizati (IQR / percentile). Nu au fost eliminati automat, deoarece pot reprezenta situatii reale de functionare.
- S-a mentinut consistenta intervalelor pentru fiecare feature.

### 4.2 Transformarea caracteristicilor

- S-a aplicat scalare de tip **StandardScaler** (media 0, deviatie standard 1).
- IMPORTANT: `scaler` este fit-uit **doar pe train**, apoi aplicat pe validation si test (evitare data leakage).
- Parametrii scalarii sunt salvati in `config/scaler.pkl` pentru reproducibilitate si pentru inferenta in etapa UI.

### 4.3 Structurarea seturilor de date

- Split stratificat: **70% train / 15% validation / 15% test**
- `random_state = 42` pentru reproducibilitate
- `y` (defect) este folosita pentru stratificare

Fisiere rezultate:
- `data/train/X_train.csv`, `data/train/y_train.csv`
- `data/validation/X_val.csv`, `data/validation/y_val.csv`
- `data/test/X_test.csv`, `data/test/y_test.csv`

### 4.4 Salvarea rezultatelor preprocesarii

- Dataset curatat complet: `data/processed/dataset_clean.csv`
- Seturi split: `data/train/`, `data/validation/`, `data/test/`
- Scaler: `config/scaler.pkl`

---

## 5. Fisiere Generate in Aceasta Etapa 

**`data/raw/`**
- `dataset_auto.csv` (dataset initial)

**`data/processed/`**
- `dataset_clean.csv` (dataset curatat + standardizat)

**`data/train/`**
- `X_train.csv`, `y_train.csv`

**`data/validation/`**
- `X_val.csv`, `y_val.csv`

**`data/test/`**
- `X_test.csv`, `y_test.csv`

**`config/`**
- `scaler.pkl`

**`src/preprocessing/`**
- `procesare_dataset.py` (curatare + scalare + split)

---

## 6. Stare Etapa

- [x] Structura repository configurata
- [x] Dataset analizat (EDA realizata)
- [x] Date preprocesate
- [x] Seturi train/val/test generate (stratificat 70/15/15, random_state=42)
- [x] Scaler salvat pentru reproducibilitate (fit doar pe train)
- [x] Documentatie actualizata in README + `data/README.md`
