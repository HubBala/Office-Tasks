import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
# Load the model
model = load_model("vitals_model.keras")
scaler = joblib.load("scaler.save")
# Input from user
heart_rate = st.number_input("Heart Rate", value=80)
systolic_bp = st.number_input("Systolic BP", value=120)
diastolic_bp = st.number_input("Diastolic BP", value=80)
spo2 = st.number_input("SpO2", value=98)
resp_rate = st.number_input("Respiratory Rate", value=18)
temperature = st.number_input("Temperature", value=36.5)

# Prepare input data  as of model
input_data = np.array([[heart_rate, systolic_bp, diastolic_bp, spo2, resp_rate, temperature]])
input_data = np.repeat(input_data, 3, axis=0)

input_data = input_data.reshape((1, 3, 6))  

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Prediction:", prediction)
