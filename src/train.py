import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)


# Load processed training data
df = pd.read_csv("data/processed/train.csv")


# Separate features and target
X = df.drop(columns=["loan_status"])
y = df["loan_status"]


# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Models
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "decision_tree": DecisionTreeClassifier(
        random_state=42
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "extra_trees": ExtraTreesClassifier(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# Create output folder
os.makedirs("models", exist_ok=True)


# Train and save every model
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
