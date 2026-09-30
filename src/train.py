import os
import pickle
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)


# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


# Load processed training data
df = pd.read_csv("data/processed/train.csv")


# Separate features and target
X = df.drop(columns=["loan_status"])
y = df["loan_status"]


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=params["data"]["test_size"],
    random_state=params["data"]["random_state"],
    stratify=y
)


# Models
models = {
    "logistic_regression": LogisticRegression(
        max_iter=params["logistic_regression"]["max_iter"],
        random_state=params["logistic_regression"]["random_state"]
    ),

    "decision_tree": DecisionTreeClassifier(
        random_state=params["decision_tree"]["random_state"]
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=params["random_forest"]["n_estimators"],
        random_state=params["random_forest"]["random_state"]
    ),

    "extra_trees": ExtraTreesClassifier(
        n_estimators=params["extra_trees"]["n_estimators"],
        random_state=params["extra_trees"]["random_state"]
    ),

    "gradient_boosting": GradientBoostingClassifier(
        random_state=params["gradient_boosting"]["random_state"]
    )
}


# Create output folder
os.makedirs("models", exist_ok=True)


# Train and save models
for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)


# Save validation data
validation_df = X_val.copy()
validation_df["loan_status"] = y_val.values

validation_df.to_csv(
    "data/processed/validation.csv",
    index=False
)


print()
print("Training completed!")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Models trained:", len(models))
