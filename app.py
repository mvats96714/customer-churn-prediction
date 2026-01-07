import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("Customer Churn Prediction – Production App")

# ---------- USER INPUTS ----------
tenure = st.number_input("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 600.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

# ---------- CREATE EMPTY INPUT ----------
input_dict = {col: 0 for col in features}

# ---------- FILL NUMERIC ----------
input_dict["tenure"] = tenure
input_dict["MonthlyCharges"] = monthly
input_dict["TotalCharges"] = total

# ---------- HANDLE CATEGORICAL (ONE-HOT STYLE) ----------
def set_feature(name):
    if name in input_dict:
        input_dict[name] = 1

# Contract
if contract == "One year":
    set_feature("Contract_One year")
elif contract == "Two year":
    set_feature("Contract_Two year")

# Dependents
if dependents == "Yes":
    set_feature("Dependents_Yes")

# Device Protection
if device_protection == "Yes":
    set_feature("DeviceProtection_Yes")
elif device_protection == "No internet service":
    set_feature("DeviceProtection_No internet service")

# ---------- CREATE DF ----------
input_df = pd.DataFrame([input_dict])

# ---------- SCALE ----------
input_scaled = scaler.transform(input_df)

# ---------- PREDICT ----------
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"Customer likely to CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"Customer likely to STAY (Probability: {prob:.2f})")

