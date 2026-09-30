# Loan Approval Prediction - ML Systems Design

A lightweight machine learning project for predicting whether a loan application will be approved or rejected.

This project was developed for the **Machine Learning Systems Design (DS-4491)** course and focuses on building a reproducible machine learning workflow using **DVC**.

---

## Project Pipeline

```text
Raw Dataset
     v
Preprocessing
     v
Train Multiple Models
     v
Evaluation
     v
Model Comparison / Metrics
```

The complete workflow is managed using DVC.

---

## Dataset

Dataset source:

https://www.kaggle.com/competitions/loan-approval-prediction-cpe-232-data-models-intl2/data

The task is binary classification:

```text
Applicant Information
        v
Machine Learning Model
        v
Loan Approved / Rejected
```

---

## Project Structure

```text
loan-approval-mlsys/
¦
+-- data/
¦   +-- raw/
¦   ¦   +-- train.csv
¦   ¦   +-- test.csv
¦   ¦
¦   +-- processed/
¦       +-- train.csv
¦       +-- test.csv
¦       +-- validation.csv
¦
+-- src/
¦   +-- preprocess.py
¦   +-- train.py
¦   +-- evaluate.py
¦
+-- models/
¦   +-- logistic_regression.pkl
¦   +-- decision_tree.pkl
¦   +-- random_forest.pkl
¦   +-- extra_trees.pkl
¦   +-- gradient_boosting.pkl
¦
+-- metrics/
¦   +-- metrics.json
¦
+-- dvc.yaml
+-- dvc.lock
+-- requirements.txt
+-- README.md
```

---

## Preprocessing

The preprocessing stage:

- Loads the raw training and test datasets
- Separates the target column `loan_status`
- Removes the `id` column
- Fills missing numerical values using the median
- Fills missing categorical values using the most frequent value
- Applies one-hot encoding to categorical features
- Saves the processed datasets

Run manually:

```bash
python src/preprocess.py
```

Processed dataset sizes:

```text
Training dataset: 32000 rows
Test dataset:     8000 rows
```

---

## Model Training

Five classification models are trained:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. Gradient Boosting

The processed training dataset is divided into:

```text
80% training   = 25,600 samples
20% validation = 6,400 samples
```

A fixed `random_state=42` is used where applicable to make the results reproducible.

Run manually:

```bash
python src/train.py
```

The trained models are saved inside:

```text
models/
```

---

## Model Evaluation

All five models are evaluated on the same validation dataset using:

- Accuracy
- Precision
- Recall
- F1-score

Current results:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5458 | 0.5469 | 0.9682 | 0.6990 |
| Decision Tree | 0.5069 | 0.5477 | 0.5439 | 0.5458 |
| Random Forest | 0.5269 | 0.5520 | 0.6974 | 0.6162 |
| Extra Trees | 0.5233 | 0.5498 | 0.6890 | 0.6116 |
| Gradient Boosting | 0.5442 | 0.5476 | 0.9380 | 0.6916 |

Evaluation results are stored in:

```text
metrics/metrics.json
```

Run evaluation manually:

```bash
python src/evaluate.py
```

---

## DVC Pipeline

The complete ML pipeline is defined in `dvc.yaml`.

```text
train.csv --+
            +--> preprocess --> train --> evaluate
test.csv ---+
```

The training stage produces five model artifacts.

The evaluation stage depends on those models and produces `metrics/metrics.json`.

Reproduce the complete pipeline:

```bash
python -m dvc repro
```

View the pipeline DAG:

```bash
python -m dvc dag
```

Check pipeline status:

```bash
python -m dvc status
```

When everything is synchronized:

```text
Data and pipelines are up to date.
```

---

## DVC Metrics

View all tracked model metrics using:

```bash
python -m dvc metrics show
```

DVC reads the metrics directly from:

```text
metrics/metrics.json
```

---

## Data Versioning

Raw datasets are tracked using DVC rather than being stored directly in Git.

```bash
python -m dvc add data/raw/train.csv
python -m dvc add data/raw/test.csv
```

DVC creates metadata files such as:

```text
train.csv.dvc
test.csv.dvc
```

Git tracks the metadata while DVC manages the actual data.

---

## DVC Remote Storage

A DVC remote is configured for storing DVC-managed datasets and model artifacts.

Push DVC files:

```bash
python -m dvc push
```

Retrieve DVC files:

```bash
python -m dvc pull
```

---

## Reproducing the Project

### 1. Clone the repository

```bash
git clone https://github.com/madnan63/loan-approval-prediction.git
```

### 2. Enter the repository

```bash
cd loan-approval-prediction
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Pull DVC-managed files

```bash
python -m dvc pull
```

### 5. Reproduce the pipeline

```bash
python -m dvc repro
```

### 6. View model metrics

```bash
python -m dvc metrics show
```

---

## Technologies Used

- Python
- pandas
- scikit-learn
- DVC
- Git
- GitHub

---

## Important DVC Commands

```bash
python -m dvc init
python -m dvc add
python -m dvc repro
python -m dvc status
python -m dvc dag
python -m dvc metrics show
python -m dvc push
python -m dvc pull
```

---

## Project Objective

The primary objective of this project is to demonstrate a **reproducible and version-controlled machine learning system** rather than building a computationally expensive model.

The project demonstrates:

- Dataset versioning
- Reproducible preprocessing
- Multiple-model training
- Model comparison
- Reproducible evaluation
- Dependency tracking
- Pipeline execution
- Metrics tracking
- DVC remote storage
- Git and GitHub integration
