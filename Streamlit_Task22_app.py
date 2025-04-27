import streamlit as st
import pickle
import numpy as np

# Load the Q-table, scaler, and KMeans model
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Action mapping
action_map = {
    0: "Balanced maintaine Food Diet will be Better",
    1: "Sever condition Medication is Needed",
    2: "Normal condition Exercise will be Good "
}

# Streamlit UI
st.title("Prescriptive Treatment Plan Recommendation for Diabetes")
st.write("Enter patient details to get a recommended treatment plan.")

# User inputs
preg = st.number_input("Pregnancies", min_value=0, max_value=20)
glucose = st.number_input("Glucose", min_value=0, max_value=200)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin", min_value=0, max_value=900)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
age = st.number_input("Age", min_value=10, max_value=100)

# Prepare the input for prediction
user_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

# Scale the input
user_input_scaled = scaler.transform(user_input)

# Get the closest state using KMeans
state = kmeans.predict(user_input_scaled)[0]

# Make prediction
if st.button("Get Recommendation"):
    action = np.argmax(q_table[state])  # Choose best action (treatment) based on Q-values
    recommendation = action_map[action]
    st.success(f"✅ Recommended Treatment: **{recommendation}**")
