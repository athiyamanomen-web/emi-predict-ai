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
    age,
    gender,
    marital_status,
    education,
    monthly_salary,
    employment_type,
    years_of_employment,
    company_type,
    house_type,
    monthly_rent,
    family_size,
    dependents,
    school_fees,
    college_fees,
    travel_expenses,
    groceries_utilities,
    other_monthly_expenses,
    existing_loans,
    current_emi_amount,
    credit_score,
    bank_balance,
    emergency_fund,
    emi_scenario,
    requested_amount,
    requested_tenure,
):
    total_expenses = (
        monthly_rent
        + school_fees
        + college_fees
        + travel_expenses
        + groceries_utilities
        + other_monthly_expenses
        + current_emi_amount
    )

    disposable_income = monthly_salary - total_expenses
    expense_to_income_ratio = safe_divide(total_expenses, monthly_salary)
    emi_to_income_ratio = safe_divide(current_emi_amount, monthly_salary)
    savings_to_income_ratio = safe_divide(bank_balance, monthly_salary)
    loan_to_income_ratio = safe_divide(requested_amount, monthly_salary)
    emergency_fund_months = safe_divide(emergency_fund, total_expenses)
    dependents_ratio = safe_divide(dependents, family_size)
    income_per_year_exp = safe_divide(monthly_salary, years_of_employment + 1)
    disposable_income_ratio = safe_divide(disposable_income, monthly_salary)

    high_financial_stress = int(expense_to_income_ratio > 0.7)
    low_emergency_fund_flag = int(emergency_fund_months < 3)
    high_loan_burden_flag = int(loan_to_income_ratio > 6)

    if credit_score >= 750:
        credit_score_bucket = "Excellent"
    elif credit_score >= 700:
        credit_score_bucket = "Good"
    elif credit_score >= 650:
        credit_score_bucket = "Fair"
    else:
        credit_score_bucket = "Poor"

    if years_of_employment <= 2:
        experience_bucket = "0-2 years"
    elif years_of_employment <= 5:
        experience_bucket = "3-5 years"
    elif years_of_employment <= 10:
        experience_bucket = "6-10 years"
    elif years_of_employment <= 20:
        experience_bucket = "11-20 years"
    else:
        experience_bucket = "20+ years"

    if monthly_salary < 30000:
        salary_bucket = "Low"
    elif monthly_salary < 60000:
        salary_bucket = "Lower-Middle"
    elif monthly_salary < 100000:
        salary_bucket = "Middle"
    else:
        salary_bucket = "High"

    salary_credit_interaction = monthly_salary * credit_score
    disposable_credit_interaction = disposable_income * credit_score
    emi_dependents_interaction = emi_to_income_ratio * dependents

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
        "emi_eligibility": "Unknown",
        "total_expenses": total_expenses,
        "disposable_income": disposable_income,
        "expense_to_income_ratio": expense_to_income_ratio,
        "emi_to_income_ratio": emi_to_income_ratio,
        "savings_to_income_ratio": savings_to_income_ratio,
        "loan_to_income_ratio": loan_to_income_ratio,
        "emergency_fund_months": emergency_fund_months,
        "dependents_ratio": dependents_ratio,
        "income_per_year_exp": income_per_year_exp,
        "disposable_income_ratio": disposable_income_ratio,
        "high_financial_stress": high_financial_stress,
        "low_emergency_fund_flag": low_emergency_fund_flag,
        "high_loan_burden_flag": high_loan_burden_flag,
        "credit_score_bucket": credit_score_bucket,
        "experience_bucket": experience_bucket,
        "salary_bucket": salary_bucket,
        "salary_credit_interaction": salary_credit_interaction,
        "disposable_credit_interaction": disposable_credit_interaction,
        "emi_dependents_interaction": emi_dependents_interaction,
    }

    input_df = pd.DataFrame([input_data])

    # Reorder columns exactly as model expects
    if hasattr(reg_model, "feature_names_in_"):
        expected_cols = list(reg_model.feature_names_in_)
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_cols]

    return input_df

st.title("Maximum EMI Amount Prediction")
st.write("Estimate the maximum affordable monthly EMI based on applicant financial profile.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    monthly_salary = st.number_input("Monthly Salary", min_value=0.0, value=50000.0)
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=3.0)

with col2:
    company_type = st.selectbox("Company Type", ["Small", "Mid-size", "Large Indian", "MNC", "Startup"])
    house_type = st.selectbox("House Type", ["Own", "Family", "Rented"])
    monthly_rent = st.number_input("Monthly Rent", min_value=0.0, value=10000.0)
    family_size = st.number_input("Family Size", min_value=1, value=3)
    dependents = st.number_input("Dependents", min_value=0, value=1)
    school_fees = st.number_input("School Fees", min_value=0.0, value=0.0)
    college_fees = st.number_input("College Fees", min_value=0.0, value=0.0)

with col3:
    travel_expenses = st.number_input("Travel Expenses", min_value=0.0, value=3000.0)
    groceries_utilities = st.number_input("Groceries & Utilities", min_value=0.0, value=10000.0)
    other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0.0, value=5000.0)
    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
    current_emi_amount = st.number_input("Current EMI Amount", min_value=0.0, value=0.0)
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=700.0)
    bank_balance = st.number_input("Bank Balance", min_value=0.0, value=100000.0)

emergency_fund = st.number_input("Emergency Fund", min_value=0.0, value=50000.0)
emi_scenario = st.selectbox(
    "EMI Scenario",
    ["Personal Loan EMI", "E-commerce Shopping EMI", "Education EMI", "Vehicle EMI", "Home Appliances EMI"]
)
requested_amount = st.number_input("Requested Loan Amount", min_value=0.0, value=200000.0)
requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, value=24)

if st.button("Predict Maximum EMI"):
    try:
        input_df = build_input_df(
            age,
            gender,
            marital_status,
            education,
            monthly_salary,
            employment_type,
            years_of_employment,
            company_type,
            house_type,
            monthly_rent,
            family_size,
            dependents,
            school_fees,
            college_fees,
            travel_expenses,
            groceries_utilities,
            other_monthly_expenses,
            existing_loans,
            current_emi_amount,
            credit_score,
            bank_balance,
            emergency_fund,
            emi_scenario,
            requested_amount,
            requested_tenure,
        )

        prediction = reg_model.predict(input_df)[0]
        st.success("Prediction completed successfully.")
        st.info(f"💰 Maximum Affordable EMI: ₹ {prediction:,.0f} per month")
        st.caption("Prediction is based on financial capacity, credit score, and expense patterns.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")