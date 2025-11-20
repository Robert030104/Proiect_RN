import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# citire dataset
data = pd.read_csv("dataset_masini.csv")

# separare intrari / etichete
X = data.drop("defect", axis=1).values
y = data["defect"].values

# scalare
scaler = StandardScaler()
X = scaler.fit_transform(X)

# impartire train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
