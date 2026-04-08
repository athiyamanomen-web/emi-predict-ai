import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("Model Performance Summary")

st.subheader("Best Selected Models")
st.markdown("""
- **Classification:** Gradient Boosting Classifier  
- **Regression:** Random Forest Regressor  
""")

st.subheader("Classification Results")
classification_results = pd.DataFrame({
    "Model": ["Gradient Boosting", "Random Forest", "Logistic Regression"],
    "Accuracy": [0.9439, 0.8736, 0.8119],
    "F1 Score": [0.9244, 0.8900, 0.8577],
    "ROC-AUC": [0.9920, 0.9775, 0.9693]
})
st.dataframe(classification_results)

st.subheader("Regression Results")
regression_results = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "Decision Tree", "Linear Regression"],
    "RMSE": [950.7997, 1090.4956, 1338.4652, 3290.9839],
    "MAE": [376.6882, 535.6075, 579.9197, 2261.5357],
    "R2": [0.9853, 0.9807, 0.9709, 0.8242],
    "MAPE": [7.0806, 22.9104, 10.2307, 128.7544]
})
st.dataframe(regression_results)

st.success(
    "MLflow was used for experiment tracking, model comparison, and final model selection."
)