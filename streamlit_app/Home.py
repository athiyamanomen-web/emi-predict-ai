import streamlit as st

st.set_page_config(page_title="EMI Predict AI", layout="wide")

st.title("💳 EMI Predict AI")
st.write("An end-to-end Machine Learning application for EMI eligibility and affordability prediction.")

st.markdown("""
### 🚀 Application Features

- **EMI Eligibility Prediction**
- **Maximum Affordable EMI Prediction**
- **Interactive Data Exploration**
- **Model Performance Dashboard**
- **Administrative Data Overview**
""")

st.subheader("📊 End-to-End Project Workflow")

st.markdown("""
1. Data Cleaning  
2. Feature Engineering  
3. Exploratory Data Analysis  
4. Classification Modeling  
5. Regression Modeling  
6. MLflow Experiment Tracking  
7. Streamlit Application Development  
8. Cloud Deployment  
""")

st.subheader("🏆 Final Selected Models")

st.markdown("""
- **Classification:** Gradient Boosting Classifier  
- **Regression:** Random Forest Regressor  
""")

st.success("Use the sidebar to navigate between different sections of the application.")