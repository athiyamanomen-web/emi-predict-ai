import os
import json
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
INPUT_PATH = "data/processed/emi_cleaned_final.csv"
OUTPUT_PATH = "data/processed/emi_featured_final.csv"
REPORT_PATH = "data/processed/feature_engineering_report.json"


# =========================================================
# HELPERS
# =========================================================
def ensure_output_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def safe_divide(numerator, denominator):
    """Safe division that returns 0 where denominator is 0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.where(denominator != 0, numerator / denominator, 0)


# =========================================================
# LOAD
# =========================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"Cleaned data loaded: {df.shape}")
    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {
        "rows_before": int(len(df)),
        "columns_before": int(df.shape[1]),
    }

    # -----------------------------------------------------
    # 1. TOTAL MONTHLY EXPENSES
    # -----------------------------------------------------
    df["total_expenses"] = (
        df["monthly_rent"]
        + df["school_fees"]
        + df["college_fees"]
        + df["travel_expenses"]
        + df["groceries_utilities"]
        + df["other_monthly_expenses"]
        + df["current_emi_amount"]
    )

    # -----------------------------------------------------
    # 2. DISPOSABLE INCOME
    # -----------------------------------------------------
    df["disposable_income"] = df["monthly_salary"] - df["total_expenses"]

    # -----------------------------------------------------
    # 3. FINANCIAL RATIOS
    # -----------------------------------------------------
    df["expense_to_income_ratio"] = safe_divide(df["total_expenses"], df["monthly_salary"])
    df["emi_to_income_ratio"] = safe_divide(df["current_emi_amount"], df["monthly_salary"])
    df["savings_to_income_ratio"] = safe_divide(df["bank_balance"], df["monthly_salary"])
    df["loan_to_income_ratio"] = safe_divide(df["requested_amount"], df["monthly_salary"])

    # -----------------------------------------------------
    # 4. EMERGENCY FUND STRENGTH
    # -----------------------------------------------------
    df["emergency_fund_months"] = safe_divide(df["emergency_fund"], df["total_expenses"])

    # -----------------------------------------------------
    # 5. DEPENDENTS PRESSURE
    # -----------------------------------------------------
    df["dependents_ratio"] = safe_divide(df["dependents"], df["family_size"])

    # -----------------------------------------------------
    # 6. INCOME STABILITY / EXPERIENCE
    # -----------------------------------------------------
    df["income_per_year_exp"] = safe_divide(df["monthly_salary"], df["years_of_employment"] + 1)

    # -----------------------------------------------------
    # 7. DISPOSABLE INCOME RATIO
    # -----------------------------------------------------
    df["disposable_income_ratio"] = safe_divide(df["disposable_income"], df["monthly_salary"])

    # -----------------------------------------------------
    # 8. EMI CAPACITY GAP
    # -----------------------------------------------------
    df["emi_capacity_gap"] = df["max_monthly_emi"] - df["current_emi_amount"]

    # -----------------------------------------------------
    # 9. REQUESTED EMI ESTIMATE
    # -----------------------------------------------------
    df["requested_emi_estimate"] = safe_divide(df["requested_amount"], df["requested_tenure"])

    # -----------------------------------------------------
    # 10. REQUESTED EMI TO INCOME
    # -----------------------------------------------------
    df["requested_emi_to_income_ratio"] = safe_divide(df["requested_emi_estimate"], df["monthly_salary"])

    # -----------------------------------------------------
    # 11. RISK FLAGS
    # -----------------------------------------------------
    df["high_financial_stress"] = (df["expense_to_income_ratio"] > 0.7).astype(int)
    df["low_emergency_fund_flag"] = (df["emergency_fund_months"] < 3).astype(int)
    df["high_loan_burden_flag"] = (df["loan_to_income_ratio"] > 6).astype(int)

    # -----------------------------------------------------
    # 12. CREDIT SCORE BUCKET
    # -----------------------------------------------------
    def credit_bucket(score):
        if score >= 750:
            return "Excellent"
        if score >= 700:
            return "Good"
        if score >= 650:
            return "Fair"
        return "Poor"

    df["credit_score_bucket"] = df["credit_score"].apply(credit_bucket)

    # -----------------------------------------------------
    # 13. EXPERIENCE BUCKET
    # -----------------------------------------------------
    def experience_bucket(years):
        if years <= 2:
            return "0-2 years"
        if years <= 5:
            return "3-5 years"
        if years <= 10:
            return "6-10 years"
        if years <= 20:
            return "11-20 years"
        return "20+ years"

    df["experience_bucket"] = df["years_of_employment"].apply(experience_bucket)

    # -----------------------------------------------------
    # 14. SALARY BUCKET
    # -----------------------------------------------------
    def salary_bucket(salary):
        if salary < 30000:
            return "Low"
        if salary < 60000:
            return "Lower-Middle"
        if salary < 100000:
            return "Middle"
        return "High"

    df["salary_bucket"] = df["monthly_salary"].apply(salary_bucket)

    # -----------------------------------------------------
    # 15. INTERACTION FEATURES
    # -----------------------------------------------------
    df["salary_credit_interaction"] = df["monthly_salary"] * df["credit_score"]
    df["disposable_credit_interaction"] = df["disposable_income"] * df["credit_score"]
    df["emi_dependents_interaction"] = df["emi_to_income_ratio"] * df["dependents"]

    created_features = [
        "total_expenses",
        "disposable_income",
        "expense_to_income_ratio",
        "emi_to_income_ratio",
        "savings_to_income_ratio",
        "loan_to_income_ratio",
        "emergency_fund_months",
        "dependents_ratio",
        "income_per_year_exp",
        "disposable_income_ratio",
        "emi_capacity_gap",
        "requested_emi_estimate",
        "requested_emi_to_income_ratio",
        "high_financial_stress",
        "low_emergency_fund_flag",
        "high_loan_burden_flag",
        "credit_score_bucket",
        "experience_bucket",
        "salary_bucket",
        "salary_credit_interaction",
        "disposable_credit_interaction",
        "emi_dependents_interaction",
    ]

    report["rows_after"] = int(len(df))
    report["columns_after"] = int(df.shape[1])
    report["created_features"] = created_features

    return df, report


# =========================================================
# SAVE
# =========================================================
def save_outputs(df: pd.DataFrame, report: dict, output_path: str, report_path: str) -> None:
    ensure_output_dir(output_path)
    ensure_output_dir(report_path)

    df.to_csv(output_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\nFeatured dataset saved to: {output_path}")
    print(f"Feature engineering report saved to: {report_path}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    df = load_data(INPUT_PATH)

    print("\nInitial shape:", df.shape)

    featured_df, report = create_features(df)

    print("\nFinal shape:", featured_df.shape)
    print("\nCreated features:")
    for col in report["created_features"]:
        print(f"- {col}")

    save_outputs(featured_df, report, OUTPUT_PATH, REPORT_PATH)
