import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
try:
    model = load_model("heart_disease_model.keras")
except FileNotFoundError:
    st.error("Error: 'model.keras' not found. Please ensure the model file is in the same directory.")
    st.stop()

try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Error: 'scaler.save' not found. Please ensure the scaler file is in the same directory.")
    st.stop()

st.title(" Heart Disease Predictor")
st.write("Enter patient data to predict heart disease presence")

# Dropdown mappings
chest_pain_map = {"Typical Angina": 3, "Atypical": 1, "Non-Anginal pain": 2, "Asymptomatic": 0}
rest_ecg_map = {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}
slope_map = {"upsloping": 0, "Flat": 1, "downsloping": 2}
vessels_map = {"Zero": 0, "one": 1, "Two": 2, "Three": 3}
thal_map = {"normal": 0, "fixed defect": 1, "Reversable defect": 2}

# User inputs
sex_options = ["Male", "Female"]
sex = st.selectbox("Sex", sex_options)
age = st.number_input("Age", 0, 100, value=60)
resting_bp = st.number_input("Resting Blood Pressure", 0, 200, value=130)
cholesterol = st.number_input("Cholesterol", 100, 600, value=250)
fasting_blood_sugar_options = ["Lower than 120 mg/ml", "Greater than 120 mg/ml"]
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", fasting_blood_sugar_options)
rest_ecg_options = list(rest_ecg_map.keys())
rest_ecg = st.selectbox("Rest ECG", rest_ecg_options, index=0)
max_heart_rate = st.number_input("Max Heart Rate Achieved", 60, 250, value=150)
exercise_induced_angina_options = ["Yes", "No"]
exercise_induced_angina = st.selectbox("Exercise Induced Angina", exercise_induced_angina_options)
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, value=1.0)
slope_options = list(slope_map.keys())
slope = st.selectbox("Slope", slope_options, index=0)
vessels_colored_options = list(vessels_map.keys())
vessels_colored = st.selectbox("Vessels Colored by Flourosopy", vessels_colored_options, index=0)
thalassemia_options = list(thal_map.keys())
thalassemia = st.selectbox("Thalassemia", thalassemia_options, index=0)
chest_pain_type_options = list(chest_pain_map.keys())
chest_pain_type = st.selectbox("Chest Pain Type", chest_pain_type_options, index=0)

# Encoding categorical variables
encoded_input = [
    age,
    1 if sex == "Male" else 0,
    chest_pain_map[chest_pain_type],
    resting_bp,
    cholesterol,
    1 if fasting_blood_sugar == "Greater than 120 mg/ml" else 0,
    rest_ecg_map[rest_ecg],
    max_heart_rate,
    1 if exercise_induced_angina == "Yes" else 0,
    oldpeak,
    slope_map[slope],
    vessels_map[vessels_colored],
    thal_map[thalassemia]
]

# Convert to array and scale
input_array = np.array([encoded_input])
input_scaled = scaler.transform(input_array)

# Predict button
if st.button("Predict"):
    prediction_probability = model.predict(input_scaled)[0][0]
    threshold = 0.5
    st.write(f"Predicted Probability: {prediction_probability:.4f}")
    if prediction_probability >= threshold:
        st.error(f"High chance of Heart Disease (Positive - Probability >= {threshold})")
    else:
        st.success(f" Low chance of Heart Disease (Negative - Probability < {threshold})")