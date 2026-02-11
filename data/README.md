# 📘 README – Data

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pintea Robert Stefan  
**Link Repository GitHub:** https://github.com/Robert030104/Proiect_RN  
**Data:** 12/02/2026  

Acest dataset este utilizat în cadrul proiectului de Rețele Neuronale (RN) – POLITEHNICA București, FIIR.

Scopul datasetului este antrenarea unui model de mentenanță predictivă auto care estimează probabilitatea apariției unui defect tehnic pe baza parametrilor specifici vehiculului și senzorilor.

## Sursa Datelor

Datasetul este generat sintetic printr-un model logistic de risc care simulează comportamentul real al unui vehicul.

✔ Minimum 40% din date sunt generate original în cadrul proiectului.  
✔ Distribuția defectelor este controlată pentru realism.  
✔ Parametrii sunt inspirați din sisteme OBD.

## Structura Datelor

Fiecare rând reprezintă o instanță a unui vehicul la un moment dat.

### Features

| Feature | Descriere |
|----------|------------|
| km | Kilometraj total |
| vechime_ani | Vechimea vehiculului |
| coolant_temp | Temperatura lichidului de răcire (°C) |
| oil_temp | Temperatura uleiului (°C) |
| oil_pressure | Presiunea uleiului |
| MAF | Debit masic aer |
| MAP | Presiune în galeria de admisie |
| battery_v | Tensiunea bateriei |
| vibratii_relanti | Nivel vibrații la relanti |
| zile_de_la_ultima_revizie | Număr zile de la ultima revizie |

---

## Variabila țintă

| Variabilă | Descriere |
|-----------|------------|
| defect | 0 = funcționare normală, 1 = risc defect |

---

## ⚖ Distribuția Claselor

- ~75% funcționare normală
- ~25% defect

Distribuția este realizată folosind un offset logistic pentru a evita dezechilibrul extrem al claselor.

---

## Modelul de Generare

Probabilitatea defectului este calculată folosind un model logistic:

p(defect) = sigmoid(w1*x1 + w2*x2 + ... + wn*xn - offset)

unde:
- ponderile reflectă importanța semnalelor critice
- offset-ul controlează rata finală a defectelor (~24–25%)

---

## Preprocesare

În etapa de procesare:

- Se normalizează datele (StandardScaler)
- Se separă în:
  - train
  - validation
  - test
- Se păstrează random_state=42 pentru reproductibilitate

---

## Scopul Datasetului

- Antrenarea unui model RN (MLP)
- Minimizarea alarmelor false
- Maximizarea recall-ului pentru clasa defect
- Obținerea unei acurateți >75%

---


## Observații

Datasetul este sintetic, dar calibrat pentru:
- realism industrial
- corelații logice între parametri
- simularea degradării progresive a componentelor



