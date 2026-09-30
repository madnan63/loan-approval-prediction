import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Create output folders if needed
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# Save trained model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)


# Save validation data for evaluation stage
validation_df = X_val.copy()
validation_df["loan_status"] = y_val.values

validation_df.to_csv(
    "data/processed/validation.csv",
    index=False
)


print("Training completed!")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))