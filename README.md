# 💳 EMI Predict AI

An end-to-end Machine Learning project to predict **EMI eligibility** and estimate **maximum affordable EMI** using classification and regression models.

---

## 🚀 Project Overview

This application helps evaluate whether a person is eligible for EMI and predicts the maximum EMI they can afford based on their financial profile.

The project includes:

* Data preprocessing and cleaning
* Feature engineering
* Exploratory Data Analysis (EDA)
* Classification and regression modeling
* MLflow experiment tracking
* Streamlit web application deployment

---

## 🧠 Machine Learning Models

### 🔹 Classification (EMI Eligibility)

* Gradient Boosting Classifier (Best Model)
* Random Forest
* Logistic Regression

### 🔹 Regression (EMI Amount)

* Random Forest Regressor (Best Model)
* Gradient Boosting
* Decision Tree
* Linear Regression

---

## 📊 Features Used

* Income and employment details
* Expenses and financial obligations
* Credit score and bank balance
* Derived ratios:

  * Expense-to-Income Ratio
  * EMI-to-Income Ratio
  * Loan-to-Income Ratio
  * Emergency Fund Months
* Interaction features and risk flags

---

## 🧩 Project Structure

```
EMIPredict-AI/
│
├── data/
│   └── processed/
│       └── emi_featured_small.csv
│
├── models/
│   ├── best_classification_model.pkl
│   └── best_regression_model.pkl
│
├── notebooks/
│   ├── EDA_Analysis.ipynb
│   ├── classification_models.ipynb
│   └── regression_models.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   └── feature_engineering.py
│
├── streamlit_app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Eligibility_Prediction.py
│       ├── 2_EMI_Amount_Prediction.py
│       ├── 3_Data_Explorer.py
│       ├── 4_Model_Performance.py
│       └── 5_Admin_Panel.py
│
├── requirements.txt
└── README.md
```

---

## 📈 Model Performance

### Classification

* Accuracy: **94.39%**
* F1 Score: **0.9244**
* ROC-AUC: **0.9920**

### Regression

* R² Score: **0.9853**
* RMSE: **950.79**
* MAE: **376.68**

---

## 🖥️ Streamlit Application Features

* EMI Eligibility Prediction
* EMI Amount Prediction
* Interactive Data Explorer
* Model Performance Dashboard
* Administrative Panel

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/athiyamanomen-web/emi-predict-ai.git
cd emi-predict-ai
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run streamlit_app/Home.py
```

---

## 📌 Important Note

* The dataset used in the app is a **reduced sample (`emi_featured_small.csv`)** for deployment.
* Full dataset is not included due to GitHub size limits.
* File paths are configured using relative paths for portability.

---

## 🧪 MLflow Integration

MLflow was used for:

* Experiment tracking
* Model comparison
* Selecting best-performing models

---

## 👨‍💻 Author

Athiyaman.P

---

## 🌟 Project Highlights

* End-to-end ML pipeline
* Real-world financial problem
* Clean deployment-ready architecture
* Production-style Streamlit app

---
