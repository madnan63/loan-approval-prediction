import json
import os
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load validation data
df = pd.read_csv("data/processed/validation.csv")

X_val = df.drop(columns=["loan_status"])
y_val = df["loan_status"]


# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)


# Make predictions
y_pred = model.predict(X_val)


# Calculate metrics
metrics = {
    "accuracy": accuracy_score(y_val, y_pred),
    "precision": precision_score(y_val, y_pred),
    "recall": recall_score(y_val, y_pred),
    "f1_score": f1_score(y_val, y_pred)
}


# Create metrics directory
os.makedirs("metrics", exist_ok=True)


# Save metrics
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


print("Evaluation completed!")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
