"""
rebuild_model.py
Rebuilds model.pkl using the local Python environment so it is compatible
with the installed scikit-learn / Python version.
"""

import sys
import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATASET_URL  = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)
DATASET_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_FILE   = "model.pkl"

# ── 1. Download dataset if not present ───────────────────────────────────────
if not os.path.exists(DATASET_FILE):
    print("Downloading dataset from GitHub ...")
    try:
        urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
        print("  [OK] Downloaded")
    except Exception as exc:
        sys.exit("  [FAIL] Could not download dataset: " + str(exc))
else:
    print("Using existing dataset: " + DATASET_FILE)

# ── 2. Load & clean ───────────────────────────────────────────────────────────
df = pd.read_csv(DATASET_FILE)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
print(f"Clean dataset shape: {df.shape}")

# ── 3. Pre-process ────────────────────────────────────────────────────────────
df_ml = df.drop("customerID", axis=1)
df_ml["Churn"] = df_ml["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

categorical_cols = df_ml.select_dtypes(include="object").columns
numerical_cols   = (
    df_ml.select_dtypes(include=["int64", "float64"])
         .columns.drop("Churn", errors="ignore")
)

df_ml = pd.get_dummies(df_ml, columns=categorical_cols, drop_first=True)

X = df_ml.drop("Churn", axis=1)
y = df_ml["Churn"]

# Scale only the original numerical columns
actual_numerical_cols = [c for c in numerical_cols if c in X.columns]
print(f"Numerical cols scaled : {actual_numerical_cols}")

scaler = StandardScaler()
X[actual_numerical_cols] = scaler.fit_transform(X[actual_numerical_cols])

print(f"Feature shape         : {X.shape}")
print("Feature columns:", list(X.columns))

# Print scaling params so app.py constants can be verified
means = X[actual_numerical_cols].mean().round(4).to_dict()
stds  = X[actual_numerical_cols].std().round(4).to_dict()
print("\nScaling params (already applied — these are post-scale means/stds, should be ~0 / ~1):")
print("  means:", means)
print("  stds :", stds)

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

# ── 5. Train Logistic Regression ──────────────────────────────────────────────
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── 6. Save model ─────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_FILE)
print("[OK] model.pkl saved  (Python " + sys.version.split()[0] + ")")
