import streamlit as st
import pickle
import pandas as pd
import json

# Load model and known therapies
model = pickle.load(open('svd_model.pkl', 'rb'))
df = pd.read_csv('patient_therapy_success.csv')
known_therapies = df['therapy_id'].unique().tolist()

# Load therapy names from JSON
with open('datasetB_sample.json', 'r') as f:
    data = json.load(f)

# Create a mapping from therapy_id to therapy_name
therapy_map = {therapy['id']: therapy['name'] for therapy in data['Therapies']}

st.title("Therapy Recommendation Based on Patient History")

st.subheader("Enter Patient History")
age = st.number_input("Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", ["male", "female", "other"])
default_text = "e.g., Abdominal aortic aneurysm or Acne"
conditions = st.text_area("Known Conditions (comma separated)", value= default_text, height=150)
tried_therapies = st.multiselect("Therapies Already Tried", known_therapies)
ratings = {}

for t in tried_therapies:
    display_name = therapy_map.get(t, t)
    ratings[t] = st.slider(f"Success Rating for {display_name}", min_value=0, max_value=100, value=50)

if st.button("Recommend Therapies"):
    # Simulate patient interactions
    interactions = []
    for therapy, success in ratings.items():
        interactions.append(('new_user', therapy, success))

    # Predict for unseen therapies
    seen_therapies = list(ratings.keys())
    unseen_therapies = [t for t in known_therapies if t not in seen_therapies]

    predictions = []
    for therapy_id in unseen_therapies:
        pred = model.predict(uid='new_user', iid=therapy_id, r_ui=None)
        predictions.append((therapy_id, pred.est))

    top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    st.subheader("Top Therapy Recommendations")
    for therapy, score in top_predictions:
        therapy_name = therapy_map.get(therapy, therapy)  # fallback to ID if name not found
        st.write(f"**{therapy_name}** — Predicted Success: {score:.2f}")
