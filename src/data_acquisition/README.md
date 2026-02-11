# 📘 README – Data_acquisition

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pintea Robert Stefan  
**Link Repository GitHub:** https://github.com/Robert030104/Proiect_RN  
**Data:** 12/02/2026 


## Descriere Generala

Acest modul este responsabil pentru generarea datelor originale utilizate
in proiectul de Retele Neuronale pentru predictia defectelor auto.

Datele sunt sintetice, dar construite pe baza unor principii fizice si logice
realiste (OBD-like signals), pentru a simula un sistem de mentenanta predictiva.

Acest modul asigura contributia originala de minimum 40% din dataset,
conform cerintelor proiectului (Etapa 4).

---

## Scop Modul

✔ Generare date numerice tip senzori auto  
✔ Simulare relatii fizice intre variabile  
✔ Generare eticheta defect pe baza model logistic de risc  
✔ Export dataset in format CSV  
✔ Control asupra ratei de defect (~24-25%)

---

## Script Principal

dataset.py


Rol:
- Genereaza N observatii
- Simuleaza distributii realiste pentru:
    - km
    - vechime_ani
    - zile_de_la_ultima_revizie
    - coolant_temp
    - oil_temp
    - oil_pressure
    - maf
    - map_kpa
    - battery_v
    - vibratii_relanti
- Calculeaza scor de risc
- Aplica functie logistica
- Determina eticheta binara: defect / normal
- Salveaza datele in `data/generated/`

---

## Model de Generare a Defectului

Eticheta defect este determinata printr-un model logistic:

risk_score = combinatie ponderata a:
    - temperaturi ridicate
    - presiune ulei scazuta
    - vibratii crescute
    - service_overdue
    - baterie joasa
    - corelatii MAP/MAF

prob_defect = sigmoid(risk_score - offset)

Offset-ul este calibrat pentru a obtine o rata de defect aproximativ 24-25%.

---

## Parametri Importanti

- N (numar observatii generate)
- Offset logistic
- Zgomot gaussian
- Corelatii intre variabile
- Prag pentru eticheta (>= 0.5)

---

## Structura Output

Fisier generat:

data/raw/dataset_auto.csv


Contine:
- 10 feature-uri numerice
- coloana defect (0 / 1)

---

## Contributie Originala

Total observatii finale: 15000  
Observatii generate prin acest modul: 9000  
Procent contributie originala: 60%

Acest modul reprezinta principala sursa de date originale din proiect.

---

## Justificare Alegere Simulare

- Nu exista acces la date reale OBD
- Se doreste control asupra distributiilor
- Permite testarea robustetii modelului
- Reproductibilitate completa

---

## Rulare Script

Din root-ul proiectului:

```bash
python src/data_acquisition/generate.py



