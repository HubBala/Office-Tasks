# Task23-1.py Streamlit app for the hybrid model for the therapy recommendations

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from tensorflow.keras.models import load_model

# Load model and preprocessors
model = load_model("hybrid_nn_model.h5", compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load dataset for therapy names and conditions
with open("datasetB_sample.json", "r") as f:
    data = json.load(f)

# Therapy dictionary
df_therapies = pd.DataFrame(data["Therapies"])
therapy_dict = df_therapies.set_index("id")["name"].to_dict()

# All therapy IDs used during training
therapy_ids = list(encoders["therapy"].classes_)

# Condition dictionary
condition_dict = {cond["id"]: cond["name"] for cond in data["Conditions"]}
condition_names = list(condition_dict.values())

def recommend_therapies(patient_input, top_n=5):
    input_data = []

    for therapy_id in therapy_ids:
        row = [
            patient_input["age"],
            encoders["gender"].transform([patient_input["gender"]])[0],
            encoders["blood_group"].transform([patient_input["blood_group"]])[0],
            encoders["therapy"].transform([therapy_id])[0],
            len(patient_input["conditions"])
        ]
        input_data.append(row)

    X = scaler.transform(input_data)
    preds = model.predict(X).flatten()

    therapy_scores = list(zip(therapy_ids, preds))
    sorted_scores = sorted(therapy_scores, key=lambda x: x[1], reverse=True)

    return [(therapy_dict.get(tid, tid), round(score, 2)) for tid, score in sorted_scores[:top_n]]

# Streamlit UI
st.title("Therapy Recommender (Neural Network Based)")

age = st.slider("Age", 0, 100, 30)
gender = st.selectbox("Gender", encoders["gender"].classes_)
blood_group = st.selectbox("Blood Group", encoders["blood_group"].classes_)
selected_conditions = st.multiselect("Select Known Conditions", condition_names)

if st.button("Recommend Therapies"):
    if not selected_conditions:
        st.warning("Please select at least one known condition to get recommendations.")
    else:
        selected_condition_ids = [cid for cid, cname in condition_dict.items() if cname in selected_conditions]
        
        patient_input = {
            "age": age,
            "gender": gender,
            "blood_group": blood_group,
            "conditions": selected_condition_ids
        }

        recommendations = recommend_therapies(patient_input)

        st.subheader("Top Therapy Recommendations:")
        for therapy, score in recommendations:
            st.write(f"**{therapy}** - Predicted Success Score: {score}")
