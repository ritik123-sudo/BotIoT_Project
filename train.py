"""
train.py — BotIoT Dataset Model Training
Trains a Random Forest classifier to detect IoT botnet attacks.
Saves the trained model + feature metadata to model/model.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score
)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset/bot_iot.csv"
MODEL_PATH   = "model/model.pkl"
TARGET_COL   = "attack"          # binary: 0 = Normal, 1 = Attack
DROP_COLS    = [                  # columns not useful for training
    "pkSeqID", "stime", "ltime",
    "saddr", "daddr", "smac", "dmac",
    "soui", "doui", "sco", "dco",
    "category", "subcategory",   # keep only binary label
]
CATEGORICAL_COLS = ["flgs", "proto", "state"]
RANDOM_STATE     = 42

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"[1/5] Loading dataset from {path} …")
    df = pd.read_csv(path, low_memory=False)
    print(f"      {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame):
    print("[2/5] Preprocessing …")

    # Drop unwanted columns (only those that actually exist)
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)

    # Encode categorical columns
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Convert all columns to numeric where possible, coerce strings to NaN
    for col in df.columns:
        if df[col].dtype == object or hasattr(df[col].dtype, 'name') and 'string' in str(df[col].dtype).lower():
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop columns that are still non-numeric after coercion
    obj_cols = df.select_dtypes(exclude="number").columns.tolist()
    if obj_cols:
        print(f"      Dropping non-numeric columns: {obj_cols}")
        df = df.drop(columns=obj_cols)

    # Fill remaining NaNs with column median
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"      Final shape: {df.shape}")
    return df, encoders


def split(df: pd.DataFrame):
    print("[3/5] Splitting train / test (80 / 20) …")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"      Attack ratio (train): {y_train.mean():.2%}")
    return X_train, X_test, y_train, y_test, list(X.columns)


def train(X_train, y_train):
    print("[4/5] Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    print("      Training complete ✓")
    return clf


def evaluate(clf, X_test, y_test):
    print("[5/5] Evaluating …")
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
    return acc, auc


def save_model(clf, feature_cols, encoders, acc, auc):
    os.makedirs("model", exist_ok=True)
    payload = {
        "model":        clf,
        "feature_cols": feature_cols,
        "encoders":     encoders,
        "metrics":      {"accuracy": acc, "roc_auc": auc},
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Model saved → {MODEL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df                              = load_data(DATASET_PATH)
    df, encoders                    = preprocess(df)
    X_train, X_test, y_train, y_test, features = split(df)
    clf                             = train(X_train, y_train)
    acc, auc                        = evaluate(clf, X_test, y_test)
    save_model(clf, features, encoders, acc, auc)
    print("\n✅  Done!  Run  python app.py  to start the web interface.\n")
