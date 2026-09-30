import json
import pickle
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Load validation data
df = pd.read_csv("data/processed/validation.csv")

X_val = df.drop(columns=["loan_status"])
y_val = df["loan_status"]


# Model files
model_files = {
    "logistic_regression": "models/logistic_regression.pkl",
    "decision_tree": "models/decision_tree.pkl",
    "random_forest": "models/random_forest.pkl",
    "extra_trees": "models/extra_trees.pkl",
    "gradient_boosting": "models/gradient_boosting.pkl"
}


results = {}


# Evaluate every model
for name, path in model_files.items():

    with open(path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
        "f1_score": float(f1_score(y_val, y_pred))
    }

    results[name] = metrics


# Save all metrics
with open("metrics/metrics.json", "w") as f:
    json.dump(results, f, indent=4)


# Display results
print("Model Evaluation Results")
print("-" * 78)

print(
    f"{'Model':<22}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1 Score':>12}"
)

print("-" * 78)

for name, metrics in results.items():
    print(
        f"{name:<22}"
        f"{metrics['accuracy']:>12.4f}"
        f"{metrics['precision']:>12.4f}"
        f"{metrics['recall']:>12.4f}"
        f"{metrics['f1_score']:>12.4f}"
    )
