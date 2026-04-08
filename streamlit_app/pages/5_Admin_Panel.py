import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Admin Panel", layout="wide")

# =========================================================
# PATH SETUP (DEPLOYMENT SAFE)
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "emi_featured_small.csv"
MODEL_DIR = BASE_DIR / "models"

st.title("Administrative Data Overview")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    st.subheader("Dataset Information")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Data Types")
    dtypes_df = df.dtypes.reset_index()
    dtypes_df.columns = ["Column", "Data Type"]
    st.dataframe(dtypes_df)

else:
    st.error("Processed dataset not found.")

if MODEL_DIR.exists():
    st.subheader("Saved Models")
    st.write([f.name for f in MODEL_DIR.iterdir() if f.is_file()])
else:
    st.error("Model directory not found.")
