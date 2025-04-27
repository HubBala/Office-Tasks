import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and scaler
model = load_model("heart_disease_model.keras")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Predictor")
st.write("Enter the patient details to check the risk of heart disease.")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
rest_ecg = st.selectbox("Resting ECG", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"])
max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
ex_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])
ca = st.selectbox("Number of Vessels Colored by Flourosopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

# Encode input into model format
input_dict = {
    'age': age,
    'sex': 1 if sex == "Male" else 0,
    'resting_blood_pressure': resting_bp,
    'cholesterol': cholesterol,
    'fasting_blood_sugar': 1 if fasting_bs == "Yes" else 0,
    'max_heart_rate': max_hr,
    'oldpeak': oldpeak,
    'chest_pain_type_Atypical angina': 1 if chest_pain == "Atypical angina" else 0,
    'chest_pain_type_Non-anginal pain': 1 if chest_pain == "Non-anginal pain" else 0,
    'chest_pain_type_Typical angina': 1 if chest_pain == "Typical angina" else 0,
    'rest_ecg_ST-T wave abnormality': 1 if rest_ecg == "ST-T wave abnormality" else 0,
    'rest_ecg_left ventricular hypertrophy': 1 if rest_ecg == "left ventricular hypertrophy" else 0,
    'exercise_induced_angina_Yes': 1 if ex_angina == "Yes" else 0,
    'slope_flat': 1 if slope == "flat" else 0,
    'slope_upsloping': 1 if slope == "upsloping" else 0,
    'vessels_colored_by_flourosopy_1': 1 if ca == 1 else 0,
    'vessels_colored_by_flourosopy_2': 1 if ca == 2 else 0,
    'vessels_colored_by_flourosopy_3': 1 if ca == 3 else 0,
    'thal_fixed defect': 1 if thal == "fixed defect" else 0,
    'thal_normal': 1 if thal == "normal" else 0,
    'thal_reversible defect': 1 if thal == "reversible defect" else 0
}

# Create DataFrame
input_df = pd.DataFrame([input_dict])

# Add missing columns (from training set)
for col in scaler.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[scaler.feature_names_in_]

# Scale the input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0][0]
    if prediction > 0.5:
        st.error(f"⚠️ High Risk of Heart Disease! (Probability: {prediction:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease. (Probability: {prediction:.2f})")
