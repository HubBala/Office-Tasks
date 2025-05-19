import json
import numpy as np
import pickle
import tensorflow as tf

# Load encoders and model only once
with open("models/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

le_gender = encoders["gender"]
le_blood = encoders["blood_group"]
le_therapy = encoders["therapy"]

with open("models/Therapy_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = tf.keras.models.load_model("models/hybrid_nn_model.h5")

# Global dictionary for mapping therapy IDs to names
with open("models/datasetB_sample.json", "r") as f:
    dataset = json.load(f)

therapy_id_to_name = {therapy["id"]: therapy["name"] for therapy in dataset.get("Therapies", [])}


def get_all_conditions():
    """Read all unique conditions from the dataset JSON file."""
    condition_list = dataset.get("Conditions", [])
    condition_set = set()

    for condition in condition_list:
        if isinstance(condition, dict):
            name = condition.get("name")
            if name:
                condition_set.add(name)
        else:
            condition_set.add(condition)

    return sorted(condition_set)


def get_options():
    """Return dropdown options: genders, blood groups, and conditions."""
    return list(le_gender.classes_), list(le_blood.classes_), get_all_conditions()


def predict_therapy_success(age, gender, blood_group, selected_conditions):
    """
    Predict top 5 therapy success scores based on input features.
    Returns: dict of {therapy name: score}
    """
    try:
        if not isinstance(age, (int, float)) or age <= 0:
            raise ValueError("Invalid age.")

        if gender not in le_gender.classes_:
            raise ValueError("Invalid gender.")

        if blood_group not in le_blood.classes_:
            raise ValueError("Invalid blood group.")

        condition_count = len(selected_conditions)

        gender_enc = le_gender.transform([gender])[0]
        blood_enc = le_blood.transform([blood_group])[0]

        therapy_scores = {}

        for therapy_id in le_therapy.classes_:
            therapy_enc = le_therapy.transform([therapy_id])[0]

            # Prepare input features
            X_input = np.array([[age, gender_enc, blood_enc, therapy_enc, condition_count]])
            X_scaled = scaler.transform(X_input)

            # Predict success score
            pred = model.predict(X_scaled, verbose=0)
            score = float(pred[0][0])

            # Map ID to name (fallback to ID if name missing)
            name = therapy_id_to_name.get(therapy_id, therapy_id)
            therapy_scores[name] = score

        # Sort and return top 5
        sorted_therapies = sorted(therapy_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_therapies[:5])

    except Exception as e:
        print("Error in prediction:", e)
        return {"Error": str(e)}
