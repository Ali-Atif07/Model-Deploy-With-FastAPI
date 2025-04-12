from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import os

X = []
Y = []

for height in range(140 ,200, 5):
    for weight in range(40 ,100 ,5):
        bmi = weight / ((height / 100) ** 2)
        label = 1 if 18.5 <= bmi <= 24.9 else 0
        X.append([height, weight])
        Y.append(label)

X = np.array(X)
Y = np.array(Y)

model = LogisticRegression()
model.fit(X, Y)

save_path = os.path.join("advance-python-pj", "Model-Deploy-FastAPI", "health_model.pkl")
joblib.dump(model, save_path)
print("Improved model saved!")