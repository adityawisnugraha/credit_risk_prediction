# Auto-install requirements if missing (good for local dev)
import os
import sys
import subprocess

# Install requirements.txt before importing anything else
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
except Exception as e:
    print(f"Failed to install requirements: {e}")

import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('model_rf.pkl')  # Ganti path jika diperlukan

# Judul aplikasi
st.title("Loan Default Prediction App")

# Input user
loan_amnt = st.number_input('Loan Amount (USD)', min_value=0, value=10000)
loan_grade = st.selectbox('Loan Grade (A-G)', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_int_rate = st.number_input('Loan Interest Rate (%)', min_value=0.0, value=10.0)
cb_person_cred_hist_length = st.number_input('Credit History Length (years)', min_value=0, value=5)
person_income = st.number_input('Person Income (USD)', min_value=0, value=50000)
person_emp_length = st.number_input('Employment Length (years)', min_value=0, value=2)
person_age = st.number_input('Person Age (years)', min_value=18, value=30)

# Tombol prediksi
if st.button("Predict"):
    # Feature engineering
    loan_percent_income = loan_amnt * person_income
    int_rate_bucket = 0 if loan_int_rate < 9.38 else 1 if loan_int_rate < 12.67 else 2
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    loan_grade_encoded = grade_map[loan_grade]
    log_income = np.log(person_income + 1)  # +1 untuk menghindari log(0)
    income_bucket = 0 if person_income < 44004 else 1 if person_income < 70000 else 2
    income_x_credit_history = person_income * cb_person_cred_hist_length
    loan_intensity = loan_amnt / (cb_person_cred_hist_length + 1e-5)  # Hindari pembagian dengan nol

    # Predict
    features = [[
        loan_percent_income, loan_grade_encoded, loan_int_rate, int_rate_bucket,
        log_income, person_income, income_bucket,
        income_x_credit_history, loan_intensity, loan_amnt,
        person_emp_length, person_age, cb_person_cred_hist_length
    ]]

    prediction_proba = model.predict_proba(features)
    best_threshold = 0.37
    prediction = (prediction_proba[:, 1] >= best_threshold).astype(int)

    class_convert = {1: 'Default', 0: 'Non-Default'}
    result = class_convert[prediction[0]]

    # Tampilkan hasil
    st.markdown(f"**Prediction:** {result}")
    st.markdown(f"**Probability of Default:** {prediction_proba[0][1]:.2f}")
