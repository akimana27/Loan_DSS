"""
WEEK 2 — Data Preprocessing & Feature Engineering
Fitted to the REAL dataset columns:
  Age, Income, LoanAmount, CreditScore, YearsExperience,
  Gender, Education, City, EmploymentType  →  LoanApproved (0/1)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib, os

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

CATEGORICAL_COLS = ['Gender', 'Education', 'City', 'EmploymentType']
NUMERICAL_COLS   = ['Age', 'Income', 'LoanAmount', 'CreditScore',
                    'YearsExperience', 'LoanToIncome']
TARGET_COL       = 'LoanApproved'


def preprocess(df: pd.DataFrame, fit: bool = True,
               encoders: dict = None, scaler: StandardScaler = None):
    df = df.copy()

    # 1. Fix bad income values (negative incomes → 0)
    df['Income'] = df['Income'].clip(lower=0)

    # 2. Feature engineering: Loan-to-Income ratio
    df['LoanToIncome'] = df['LoanAmount'] / df['Income'].replace(0, np.nan)

    # 3. Impute missing values
    for col in ['Income', 'CreditScore', 'LoanToIncome']:
        df[col] = df[col].fillna(df[col].median())
    df['Education'] = df['Education'].fillna('Unknown')

    # 4. Encode categoricals
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    features = NUMERICAL_COLS + CATEGORICAL_COLS
    y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None
    X = df[features]

    # 5. Scale
    if fit:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=features)

    return X_scaled, y, encoders, scaler


def load_and_split(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows | {df.shape[1]} columns")
    X, y, encoders, scaler = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.pkl'))
    joblib.dump(scaler,   os.path.join(MODELS_DIR, 'scaler.pkl'))
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Class balance — 0:{(y_train==0).sum()} | 1:{(y_train==1).sum()}")
    return X_train, X_test, y_train, y_test, encoders, scaler


def preprocess_single(input_dict: dict):
    encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.pkl'))
    scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    df = pd.DataFrame([input_dict])
    X, _, _, _ = preprocess(df, fit=False, encoders=encoders, scaler=scaler)
    return X


if __name__ == '__main__':
    load_and_split('data/loan_data_raw.csv')
    print("Preprocessing complete.")