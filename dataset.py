import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

km = np.random.randint(10000, 300000, n)
temperatura = np.random.normal(90, 15, n).clip(60, 140)
erori_obd = np.random.poisson(1, n)
consum = np.random.normal(7, 2, n).clip(3, 20)
vibratii = np.random.normal(0.5, 0.2, n).clip(0.1, 2)
vechime = np.random.randint(1, 20, n)
tip_motor = np.random.randint(0, 2, n)

defect = (
    (km > 200000).astype(int) +
    (temperatura > 110).astype(int) +
    (erori_obd >= 3).astype(int) +  
    (vibratii > 1).astype(int)
)

defect = (defect > 1).astype(int)

data = pd.DataFrame({
    "km": km,
    "temperatura_motor": temperatura,
    "nr_erori_obd": erori_obd,
    "consum": consum,
    "vibratii": vibratii,
    "vechime_ani": vechime,
    "tip_motor": tip_motor,
    "defect": defect
})

data.to_csv("dataset_masini.csv", index=False)

print("Dataset generat cu succes!")
