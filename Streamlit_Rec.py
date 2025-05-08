# import streamlit as st 
# import pickle
# import numpy as np 

# # Load the Q-table
# with open("q_table.pkl", "rb") as f:
#     q_table = pickle.load(f)

# # Load the scaler
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# # Action mapping
# action_map = {
#     0: "Diet",
#     1: "Medication",
#     2: "Exercise"
# }

# # Streamlit UI
# st.title("Recommendations for the Prescriptive Analysis of Diabetes")
# st.write("Input Patient Details to Get a Recommended Treatment Plan.")

# # User inputs
# preg = st.number_input("Pregnancies", min_value=0, max_value=20)
# glucose = st.number_input("Glucose", min_value=0, max_value=200)
# bp = st.number_input("Blood Pressure", min_value=0, max_value=150)
# skin = st.number_input("Skin Thickness", min_value=0, max_value=100)
# insulin = st.number_input("Insulin", min_value=0, max_value=900)
# bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
# dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
# age = st.number_input("Age", min_value=10, max_value=100)

# # Prepare the input
# user_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

# # Scale the input using the loaded scaler
# try:
#     user_input_scaled = scaler.transform(user_input)
#     state = tuple(user_input_scaled[0].round(1))  # round for stable lookup
# except Exception as e:
#     st.error(f"Error in scaling input: {e}")
#     st.stop()

# # Make prediction
# if st.button("Get Recommendation"):
#     if state in q_table:
#         action = np.argmax(q_table[state])
#         recommendation = action_map[action]
#         st.success(f"✅ Recommended Treatment: **{recommendation}**")
#     else:
#         st.warning("⚠️ This input pattern was not seen during training. Try changing the values.")


import streamlit as st 
import pickle
import numpy as np 

# Load Q-table and scaler
@st.cache_resource
def load_model():
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return q_table, scaler

q_table, scaler = load_model()

# Action map
action_map = {
    0: "Diet",
    1: "Medication",
    2: "Exercise"
}

# UI
st.title("Prescriptive Treatment Recommendation for Diabetes")
st.write("Enter patient data to get a suggested treatment plan.")

# Input fields
preg = st.number_input("Pregnancies", min_value=0, max_value=20)
glucose = st.number_input("Glucose", min_value=0, max_value=200)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin", min_value=0, max_value=900)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
age = st.number_input("Age", min_value=10, max_value=100)

# Input array
user_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

# Predict
if st.button("Get Recommendation"):
    try:
        scaled_input = scaler.transform(user_input)
        state = tuple(scaled_input[0].round(1))
        if state in q_table:
            action = np.argmax(q_table[state])
            st.success(f"✅ Recommended Treatment: **{action_map[action]}**")
        else:
            st.warning("⚠️ This pattern was not seen during training. Try adjusting the inputs.")
    except Exception as e:
        st.error(f"Error: {e}")
