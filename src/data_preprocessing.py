import os
import re
import json
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
INPUT_PATH = "data/raw/EMI_dataset.csv"
OUTPUT_PATH = "data/processed/emi_cleaned_final.csv"
REPORT_PATH = "data/processed/preprocessing_report.json"


# =========================================================
# HELPERS
# =========================================================
def ensure_output_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def normalize_missing_like(val):
    if pd.isna(val):
        return np.nan

    s = str(val).strip().lower()
    if s in {"", "nan", "nan.0", "none", "null", "na", "n/a"}:
        return np.nan

    return val


def fix_repeated_decimal(val):
    if pd.isna(val):
        return np.nan

    s = str(val).strip()

    if s.lower() in {"", "nan", "nan.0", "none", "null", "na", "n/a"}:
        return np.nan

    if re.fullmatch(r"\d+(\.0)+", s):
        s = re.sub(r"(\.0)+$", ".0", s)

    return s


def extract_first_numeric(val):
    if pd.isna(val):
        return np.nan

    s = str(val).strip()

    if s.lower() in {"", "nan", "nan.0", "none", "null", "na", "n/a"}:
        return np.nan

    match = re.search(r"\d+(\.\d+)?", s)
    return match.group() if match else np.nan


# =========================================================
# LOAD
# =========================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"Data loaded: {df.shape}")
    return df


# =========================================================
# PREPROCESS
# =========================================================
def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}

    report["rows_before"] = int(len(df))
    report["columns_before"] = int(df.shape[1])

    # Normalize missing-like values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(normalize_missing_like)

    # AGE
    df["age"] = (
        df["age"]
        .apply(normalize_missing_like)
        .apply(fix_repeated_decimal)
        .apply(extract_first_numeric)
    )
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df.loc[(df["age"] < 18) | (df["age"] > 70), "age"] = np.nan
    df["age"].fillna(df["age"].median(), inplace=True)

    # GENDER
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    df["gender"] = df["gender"].map({
        "male": "Male",
        "m": "Male",
        "female": "Female",
        "f": "Female"
    })

    # EDUCATION
    df["education"].fillna(df["education"].mode()[0], inplace=True)

    # MONTHLY SALARY
    df["monthly_salary"] = (
        df["monthly_salary"]
        .apply(normalize_missing_like)
        .apply(fix_repeated_decimal)
        .apply(extract_first_numeric)
    )
    df["monthly_salary"] = pd.to_numeric(df["monthly_salary"], errors="coerce")

    df.loc[(df["monthly_salary"] <= 0) | (df["monthly_salary"] > 1_000_000), "monthly_salary"] = np.nan
    df["monthly_salary"].fillna(df["monthly_salary"].median(), inplace=True)

    # RENT
    df.loc[df["house_type"] == "Own", "monthly_rent"] = 0

    rent_mode = df.groupby("house_type")["monthly_rent"].transform(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )
    df["monthly_rent"].fillna(rent_mode, inplace=True)

    # CREDIT SCORE
    df.loc[(df["credit_score"] <= 0) | (df["credit_score"] > 1000), "credit_score"] = np.nan
    df["credit_score"].fillna(df["credit_score"].median(), inplace=True)

    # BANK BALANCE
    df["bank_balance"] = (
        df["bank_balance"]
        .apply(normalize_missing_like)
        .apply(fix_repeated_decimal)
        .apply(extract_first_numeric)
    )
    df["bank_balance"] = pd.to_numeric(df["bank_balance"], errors="coerce")

    df.loc[df["bank_balance"] < 0, "bank_balance"] = np.nan
    df["bank_balance"].fillna(df["bank_balance"].median(), inplace=True)

    # EMERGENCY FUND
    df.loc[df["emergency_fund"] < 0, "emergency_fund"] = np.nan
    df["emergency_fund"].fillna(df["emergency_fund"].median(), inplace=True)

    # LOGIC FIXES
    df.loc[df["dependents"] > df["family_size"], "dependents"] = df["family_size"]
    df.loc[df["existing_loans"] == "No", "current_emi_amount"] = 0

    # CAST
    for col in ["family_size", "dependents", "requested_tenure"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    report["rows_after"] = int(len(df))
    report["columns_after"] = int(df.shape[1])

    return df.reset_index(drop=True), report


# =========================================================
# SAVE
# =========================================================
def save_outputs(df, report):
    ensure_output_dir(OUTPUT_PATH)
    ensure_output_dir(REPORT_PATH)

    df.to_csv(OUTPUT_PATH, index=False)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print("Saved cleaned dataset + report")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    df = load_data(INPUT_PATH)
    cleaned_df, report = preprocess_data(df)
    save_outputs(cleaned_df, report)
