# README – Etapa 6: Analiza Performantei, Optimizare si Concluzii Finale

**Disciplina:** Retele Neuronale  
**Institutie:** POLITEHNICA Bucuresti – FIIR  
**Student:** Pintea Robert-Stefan  
**Repository:** GitHub – Robert030104/Proiect_RN  
**Data predarii:** 12.02.2026

---

## Scopul Etapei 6

Aceasta etapa corespunde punctelor:
- **7. Analiza performantei si optimizarea parametrilor**
- **8. Analiza si agregarea rezultatelor**
- **9. Formularea concluziilor finale**

**Obiectiv principal:** Optimizarea modelului RN pentru predictia defectelor auto si finalizarea livrabilor: model optimizat salvat separat, prag optim, evaluare clara si UI actualizata care foloseste modelul optimizat.

---

## PREREQUISITE – Etapa 5 (indeplinit)

- [x] Pipeline complet functional: `dataset.py -> procesare_dataset.py -> split.py -> train_model.py -> evaluare_model.py`
- [x] Model salvat + scaler salvat
- [x] UI Streamlit functionala pentru inferenta

---

# 1) Ce s-a schimbat in Etapa 6 

### 1.1 Schimbari cheie (rezolvate practic in proiect)
1. **Dataset V5 corect calibrat ca defect rate**  
   - Rata defect finala obtinuta: **~0.23** (stabila pe train/val/test)
   - Split stratificat: train/val/test au rate aproape identice (stabilitate evaluare)

2. **Fix mismatch coloane la split**  
   - Problema: fisiere vechi `X_val.csv` / `X_validation.csv` produceau lipsa coloane
   - Solutie: `split.py` suprascrie in mod standard fisierele:
     - `data/train/X_train.csv`, `data/train/y_train.csv`
     - `data/validation/X_val.csv`, `data/validation/y_val.csv`
     - `data/test/X_test.csv`, `data/test/y_test.csv`

3. **Model optimizat salvat separat (cerinta)**  
   - `models/trained_model.pt` = model vechi (baseline / ramas pentru comparatie)
   - `models/optimized_model.pt` = model final pentru predare (Etapa 6)

4. **Optimizare prag decizie pentru cerinte minime**  
   - Prag optimizat salvat in checkpoint ca **`threshold_f1`**
   - Constrangere: `min_acc >= 0.75` la cautarea pragului

5. **UI actualizata sa incarce modelul OPTIMIZAT**  
   - UI foloseste: `models/optimized_model.pt`
   - Prag folosit: `threshold_f1` (daca exista) altfel `threshold_auto` altfel 0.5
   - Mapping corect intre campurile UI si feature-urile modelului (km, coolant_temp, etc.)

---

# 2) Rezultate obtinute (azi)

## 2.1 Antrenare model optimizat
**Output relevant:**
- `Best AUC: 0.9745`
- `Best epoch: 33`
- `Auto threshold (val, max_fpr=0.130): 0.320`

Interpretare:
- Separare foarte buna intre clase (AUC foarte mare)
- Modelul este stabil si generalizeaza bine pe split-ul stratificat

## 2.2 Prag optimizat pentru cerinte (accuracy + F1)
Rulare:
- `optimize_threshold.py` pe setul de validation

**Rezultat:**
- `best_thr=0.465`
- `f1=0.9490`
- `acc=0.9767`
- salvat in model ca: `threshold_f1`

**Status cerinte minime:**
- Accuracy >= 0.75 
- F1 >= 0.65  

---

# 3) Livrabile Etapa 6 (conform cerinte)

## 3.1 Fisiere obligatorii in repository

- `models/optimized_model.pt`  
- `models/trained_model.pt`    
- `results/final_metrics.json` 
- `docs/screenshots/confusion_matrix_optimized.png` 
- `docs/screenshots/inference_optimized.png` )
- `docs/screenshots/confusion_matrix_baseline.png`

---

# 4) Instructiuni de rulare (pipeline complet)

> Ruleaza comenzile din root-ul proiectului: `D:\Proiect_RN\Proiect_RN`

### 4.1 Generare dataset
```bash
py src/data_acquisition/dataset.py

4.2 Preprocesare + scaler
py src/preprocesing/procesare_dataset.py

4.3 Split (stratificat, suprascrie fisierele vechi)
py src/preprocesing/split.py

4.4 Train model optimizat
py src/neural_network/train_model.py

4.5 Optimizare prag (min_acc >= 0.75)
py src/neural_network/optimize_threshold.py --model models/optimized_model.pt --x data/validation/X_val.csv --y data/validation/y_val.csv --scaler scaler.pkl --min_acc 0.75

4.6 Evaluare finala (test) + salvare metrics + confusion matrix
py src/neural_network/evaluare_model.py --model models/optimized_model.pt --x data/test/X_test.csv --y data/test/y_test.csv --scaler scaler.pkl --save_cm

4.7 UI (Streamlit) – incarcare model optimizat
streamlit run src/app/app.py

5) Structura repository (final)
Proiect_RN/
├── README.md
├── README_Etapa5_Antrenare_RN.md
├── README_Etape6_Analiza_Performantei_Optimizare_Concluzii.md
├── scaler.pkl
├── data/
│   ├── raw/
│   │   └── dataset_auto.csv
│   ├── processed/
│   │   └── dataset_processed.csv
│   ├── train/
│   │   ├── X_train.csv
│   │   └── y_train.csv
│   ├── validation/
│   │   ├── X_val.csv
│   │   └── y_val.csv
│   └── test/
│       ├── X_test.csv
│       └── y_test.csv
├── models/
│   ├── trained_model.pt
│   └── optimized_model.pt
├── results/
│   └── final_metrics.json
├── docs/
│   └── screenshots/
│       ├── confusion_matrix_optimized.png
│       └── inference_optimized.png
└── src/
    ├── app/
    │   └── app.py
    ├── data_acquisition/
    │   └── dataset.py
    ├── preprocesing/
    │   ├── procesare_dataset.py
    │   └── split.py
    └── neural_network/
        ├── model.py
        ├── train_model.py
        ├── optimize_threshold.py
        └── evaluare_model.py

6) Concluzii finale (Etapa 6)

Optimizarea reala a venit din corectarea pipeline-ului + standardizarea feature-urilor
(aceleasi coloane in processed/split/train/eval/UI).

Separarea claselor este acum foarte buna, reflectata de:

ROC-AUC ~0.97+

Accuracy ~0.97+

F1 ~0.95+

Pragul este salvat in model (threshold_f1) si folosit in UI pentru o decizie consistenta.
