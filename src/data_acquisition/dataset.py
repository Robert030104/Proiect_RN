import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000  


km = np.random.randint(10000, 300000, n)

vechime_ani = np.random.randint(1, 20, n)


temperatura_motor = np.random.normal(90, 12, n).clip(60, 130)


temperatura_ulei = np.random.normal(95, 15, n).clip(60, 150)


presiune_ulei = np.random.normal(40, 8, n).clip(10, 80)


vibratii = np.random.normal(0.5, 0.25, n).clip(0.1, 3)


ore_de_la_revizie = np.random.randint(0, 600, n)


km_de_la_schimb_ulei = np.random.randint(0, 30000, n)


maf = np.random.normal(80, 30, n).clip(5, 400)


map = np.random.normal(60, 25, n).clip(20, 120)

# ------------------------------------------------
# LOGICA DEFECT
# ------------------------------------------------

defect_score = (
    (km > 220000).astype(int) +
    (temperatura_motor > 110).astype(int) +
    (temperatura_ulei > 130).astype(int) +
    (presiune_ulei < 25).astype(int) +
    (vibratii > 1.2).astype(int) +
    (ore_de_la_revizie > 400).astype(int) +
    (km_de_la_schimb_ulei > 20000).astype(int) +
    (maf > 200).astype(int) +
    (map > 100).astype(int)
)

# Consideram defect daca sunt cel putin 2 probleme critice
defect = (defect_score >= 2).astype(int)

# Dataset final
data = pd.DataFrame({
    "km": km,
    "vechime_ani": vechime_ani,
    "temperatura_motor": temperatura_motor,
    "temperatura_ulei": temperatura_ulei,
    "presiune_ulei": presiune_ulei,
    "vibratii": vibratii,
    "ore_de_la_revizie": ore_de_la_revizie,
    "km_de_la_schimb_ulei": km_de_la_schimb_ulei,
    "maf": maf,
    "map": map,
    "defect": defect
})

data.to_csv("dataset_auto.csv", index=False)

print("Dataset generat cu succes!")

