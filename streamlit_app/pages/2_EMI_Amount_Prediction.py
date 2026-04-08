import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="EMI Amount Prediction", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "best_regression_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

reg_model = load_model()

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

def build_input_df(
    age, gender, marital_status, education, monthly_salary,
    employment_type, years_of_employment, company_type, house_type,
    monthly_rent, family_size, dependents, school_fees, college_fees,
    travel_expenses, groceries_utilities, other_monthly_expenses,
    existing_loans, current_emi_amount, credit_score, bank_balance,
    emergency_fund, emi_scenario, requested_amount, requested_tenure,
):
    total_expenses = (
        monthly_rent + school_fees + college_fees + travel_expenses +
        groceries_utilities + other_monthly_expenses + current_emi_amount
    )

    disposable_income = monthly_salary - total_expenses
    expense_to_income_ratio = safe_divide(total_expenses, monthly_salary)
    emi_to_income_ratio = safe_divide(current_emi_amount, monthly_salary)
    savings_to_income_ratio = safe_divide(bank_balance, monthly_salary)
    loan_to_income_ratio = safe_divide(requested_amount, monthly_salary)
    emergency_fund_months = safe_divide(emergency_fund, total_expenses)

    high_financial_stress = int(expense_to_income_ratio > 0.7)
    low_emergency_fund_flag = int(emergency_fund_months < 3)
    high_loan_burden_flag = int(loan_to_income_ratio > 6)

    input_data = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "total_expenses": total_expenses,
        "disposable_income": disposable_income,
        "expense_to_income_ratio": expense_to_income_ratio,
        "emi_to_income_ratio": emi_to_income_ratio,
        "savings_to_income_ratio": savings_to_income_ratio,
        "loan_to_income_ratio": loan_to_income_ratio,
        "emergency_fund_months": emergency_fund_months,
        "high_financial_stress": high_financial_stress,
        "low_emergency_fund_flag": low_emergency_fund_flag,
        "high_loan_burden_flag": high_loan_burden_flag,
    }

    return pd.DataFrame([input_data])

st.title("Maximum EMI Amount Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    monthly_salary = st.number_input("Monthly Salary", min_value=0.0, value=50000.0)

with col2:
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=3.0)
    company_type = st.selectbox("Company Type", ["Small", "Mid-size", "Large Indian", "MNC", "Startup"])
    house_type = st.selectbox("House Type", ["Own", "Family", "Rented"])
    monthly_rent = st.number_input("Monthly Rent", min_value=0.0, value=10000.0)

with col3:
    family_size = st.number_input("Family Size", min_value=1, value=3)
    dependents = st.number_input("Dependents", min_value=0, value=1)
    current_emi_amount = st.number_input("Current EMI Amount", min_value=0.0, value=0.0)
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=700.0)

requested_amount = st.number_input("Requested Loan Amount", min_value=0.0, value=200000.0)
requested_tenure = st.number_input("Requested Tenure", min_value=1, value=24)

if st.button("Predict EMI"):
    input_df = build_input_df(
        age, gender, marital_status, education, monthly_salary,
        employment_type, years_of_employment, company_type, house_type,
        monthly_rent, family_size, dependents, 0, 0, 0, 0, 0,
        "No", current_emi_amount, credit_score, 0, 0,
        "Personal Loan EMI", requested_amount, requested_tenure
    )

    prediction = reg_model.predict(input_df)[0]
    st.success(f"Estimated EMI: ₹ {prediction:,.0f}")
