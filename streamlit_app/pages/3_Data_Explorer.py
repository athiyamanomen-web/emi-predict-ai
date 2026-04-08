import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Data Explorer", layout="wide")

# =========================================================
# PATH SETUP (DEPLOYMENT SAFE)
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "emi_featured_small.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("Interactive Data Explorer")
st.write("Explore the processed EMI dataset interactively.")

col1, col2 = st.columns(2)

with col1:
    selected_column = st.selectbox("Select a column to inspect", df.columns)

with col2:
    rows_to_show = st.slider("Rows to preview", min_value=5, max_value=50, value=10)

st.subheader("Dataset Preview")
st.dataframe(df.head(rows_to_show))

st.subheader(f"Summary of: {selected_column}")

if df[selected_column].dtype == "object":
    st.write(df[selected_column].value_counts().head(20))
else:
    st.write(df[selected_column].describe())

st.subheader("Missing Values")
missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["Column", "Missing Count"]
st.dataframe(missing_df[missing_df["Missing Count"] > 0])

st.subheader("Target Distribution")
st.bar_chart(df["emi_eligibility"].value_counts())
