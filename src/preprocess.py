import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# Load raw data
train_df = pd.read_csv("data/raw/train.csv")
test_df = pd.read_csv("data/raw/test.csv")


# Separate target
X_train = train_df.drop(columns=["loan_status"])
y_train = train_df["loan_status"]

X_test = test_df.copy()


# Drop ID because it is only an identifier
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])


# Identify column types
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X_train.select_dtypes(include=["object"]).columns


# Numeric preprocessing
numeric_transformer = SimpleImputer(strategy="median")


# Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# Fit on training data and transform both datasets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Get processed column names
feature_names = preprocessor.get_feature_names_out()


# Convert to DataFrames
X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


# Add target back to processed training data
X_train_processed["loan_status"] = y_train.values


# Save processed datasets
X_train_processed.to_csv(
    "data/processed/train.csv",
    index=False
)

X_test_processed.to_csv(
    "data/processed/test.csv",
    index=False
)


print("Preprocessing completed!")
print("Processed train shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)