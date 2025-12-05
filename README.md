# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pintea Robert Stefan  
**Data:** 20/11/2025  


---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
Proiect_RN/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Senzori vehicule
* **Modul de achiziție:** ☐ Senzori reali / X Simulare / ☐ Fișier extern / X Generare programatică
* **Perioada / condițiile colectării:** Noiembrie 2025

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 1000
* **Număr de caracteristici (features):** 7
* **Tipuri de date:** X Numerice / ☐ Categoriale / ☐ Temporale / ☐ Imagini
* **Format fișiere:** X CSV / ☐ TXT / ☐ JSON / ☐ PNG / ☐ Altele: [...]

### 2.3 Descrierea fiecărei caracteristici

| Caracteristica        | Tip     | Unitate | Descriere                         | Domeniu valori    |
|-----------------------|---------|---------|-----------------------------------|-------------------|
| km                    | numeric | km      | Kilometraj total al vehiculului   | 10000 - 300000    |
| vechime_ani           | numeric | ani     | Varsta vehiculului                | 1 - 20            |
| temperatura_motor     | numeric | °C      | Temp. lichid racire               | 60 - 130          |
| temperatura_ulei      | numeric | °C      | Temp. ulei motor                  | 60 - 150          |
| presiune_ulei         | numeric | psi     | Presiune ulei                     | 10 - 80           |
| vibratii              | numeric | mm/s    | Nivel vibratii                    | 0.1 - 3.0         |
| ore_de_la_revizie     | numeric | ore     | Ore de la ultima revizie          | 0 - 600           |
| km_de_la_schimb_ulei  | numeric | km      | Km de la schimb ulei              | 0 - 30000         |
| maf                   | numeric | g/s     | Debit aer MAF                     | 5 - 400           |
| map                   | numeric | kPa     | Presiune MAP                      | 20 - 120          |
| defect                | numeric | 0/1     | 1 = defect, 0 = normal            | 0 sau 1           |


**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

În această etapă au fost aplicate statistici descriptive asupra variabilelor din setul de date. Pentru fiecare caracteristică numerică au fost calculate media, mediana și deviația standard, împreună cu valorile minim–maxim și quartilele (Q1, Q2, Q3). Au fost generate histograme pentru a observa distribuția datelor, iar outlierii au fost identificați utilizând metoda IQR și percentila 1%–99%. Aceste statistici oferă o imagine generală asupra comportamentului datelor înainte de antrenarea rețelei neuronale


### 3.2 Analiza calității datelor

Calitatea datelor a fost verificată prin identificarea valorilor lipsă, a tipurilor de date și a consistenței intervalelor pentru fiecare variabilă numerică. Setul de date nu conține valori lipsă, iar toate variabilele respectă tipurile așteptate. Domeniile valorilor se încadrează în limite realiste pentru parametrii unui vehicul. De asemenea, s-a verificat existența unor valori extreme care ar putea afecta procesul de antrenare.

### 3.3 Probleme identificate

Analiza a evidențiat prezența unor valori extreme (outlieri), în special la variabile precum kilometrajul, temperaturile și valorile MAF/MAP, care pot influența distribuția datelor. Deși nu există valori lipsă, anumite variabile prezintă dispersie ridicată, ceea ce poate necesita normalizare înainte de antrenare. O parte dintre valorile extreme pot proveni din comportamente reale ale vehiculului, însă unele pot reprezenta măsurători atipice ale senzorilor.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

Datasetul nu conține înregistrări duplicate.
Nu au fost identificate valori lipsă; nu a fost necesară imputarea.
Outlierii au fost identificați prin metoda IQR și analiza percentilelor (1% și 99%).
Deoarece valorile extreme pot reflecta situații tehnice reale, nu au fost eliminate automat, însă normalizarea ulterioară reduce impactul lor în procesul de antrenare.

### 4.2 Transformarea caracteristicilor

Toate variabilele numerice au fost scalate folosind media și deviația standard ale datasetului.
Encocoding-ul nu a fost necesar, deoarece datasetul conține doar variabile numerice.


### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70% – train
* 15% – validation
* 15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data\processed\dataset_clean.csv`
* Seturi train/val/test în foldere dedicate

---

##  5. Fișiere Generate în Această Etapă
 
* În urma proceselorde, au fost generate și organizate următoarele fișiere în structura proiectului:

 `data/raw/`

* Conține datasetul inițial, neprelucrat:

`dataset_auto.csv`

`data/processed/`

* Include versiunea curățată și standardizată a întregului dataset:

`dataset_clean.csv`

`data/train/, data/validation/, data/test/`

* Conțin seturile finale utilizate în antrenare, validare și testare:

`X_train.csv, y_train.csv`

`X_val.csv`, `y_val.csv`

`X_test.csv`, `y_test.csv`

`config/`

* Parametrii scalării folosiți ulterior în model:

`scaler.pkl`

`src/preprocessing/`

* Scriptul responsabil cu preprocesarea datelor:

`procesare_dataset.py`

* Documentație asociată setului de date utilizat în proiect.

---

##  6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date preprocesate
- [x] Seturi train/val/test generate
- [x] Documentație actualizată în README + `data/README.md`

---
