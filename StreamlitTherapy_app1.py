import streamlit as st
import pickle
import pandas as pd
import json

# Load the collaborative filtering model and the dataset
model = pickle.load(open('svd_model.pkl', 'rb'))
df = pd.read_csv('patient_therapy_success.csv')

# Load the therapy names and conditions from the datasetB_sample.json file
with open('datasetB_sample.json', 'r') as f:
    data = json.load(f)

# Mapping of therapy_id to therapy name
therapy_name_mapping = {therapy['id']: therapy['name'] for therapy in data['Therapies']}

# Create a dictionary to map conditions for filtering therapies
condition_name_mapping = {condition['id']: condition['name'] for condition in data['Conditions']}

# Get all therapy ids for the multiselect and dropdown
known_therapies = df['therapy_id'].unique().tolist()

# Mapping from therapy_name to therapy_id
name_to_id = {v: k for k, v in therapy_name_mapping.items()}

# Set up Streamlit UI
st.title("Therapy Recommendation Based on Patient History")

# Patient Information Form
st.subheader("Enter Patient Information")
age = st.number_input("Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", ["male", "female", "other"])
country_of_residence = st.selectbox("Country of Residence", ["USA", "India", "UK", "Other"])

# Select patient's known conditions
conditions = st.multiselect("Known Conditions", options=list(condition_name_mapping.values()))

# Display therapies already tried
st.subheader("Therapies Already Tried")
therapy_names = [therapy_name_mapping[therapy_id] for therapy_id in known_therapies]
tried_therapies = st.multiselect("Select Therapies You Have Already Tried", therapy_names)

# Collect ratings for therapies that were tried
ratings = {}
for t in tried_therapies:
    therapy_id = name_to_id[t]  # Convert therapy name back to therapy_id
    ratings[therapy_id] = st.slider(f"Success Rating for {t}", min_value=0, max_value=100, value=50)

# Button to trigger therapy recommendations
if st.button("Recommend Therapies"):
    # Simulate patient interactions for collaborative filtering
    interactions = [('new_user', therapy_id, success) for therapy_id, success in ratings.items()]

    # List of therapies that haven't been tried yet
    seen_therapies = list(ratings.keys())
    unseen_therapies = [t for t in known_therapies if t not in seen_therapies]

    # Predict success for unseen therapies
    predictions = []
    for therapy_id in unseen_therapies:
        pred = model.predict(uid='new_user', iid=therapy_id, r_ui=None)
        predictions.append((therapy_id, pred.est))

    # Sort therapies by predicted success and take top 5
    top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    # Display top therapy recommendations
    st.subheader("Top Therapy Recommendations")
    for therapy_id, score in top_predictions:
        therapy_name = therapy_name_mapping.get(therapy_id, "Unknown Therapy")
        st.write(f"**{therapy_name}** - Predicted Success: {score:.2f}")

# Optionally: Displaying the conditions and therapy types for clarity
st.sidebar.subheader("Condition Details")
for condition in conditions:
    st.sidebar.write(f"Condition: {condition}")

st.sidebar.subheader("Therapy Types")
therapy_types = {therapy['type'] for therapy in data['Therapies']}
for therapy_type in therapy_types:
    st.sidebar.write(f"- {therapy_type}")
