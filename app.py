import streamlit as st
import joblib
import pandas as pd

# Load model & scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Customer Churn Prediction App")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=800.0)

if st.button("Predict Churn"):
    input_df = pd.DataFrame([[tenure, monthly_charges, total_charges]], 
                            columns=["tenure", "MonthlyCharges", "TotalCharges"])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success("🔴 Customer Will Churn" if prediction[0] == 1 else "🟢 Customer Will Stay")
