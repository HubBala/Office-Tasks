import pickle
import numpy as np

# Load models once at import
with open("models/scalerRec.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("models/q_table_Rec.pkl", "rb") as f:
    q_table = pickle.load(f)

# Convert q_table to NumPy array if needed
q_table = np.array(q_table) if isinstance(q_table, list) else q_table

# Validate q_table shape
if not isinstance(q_table, np.ndarray) or q_table.ndim != 2:
    raise ValueError("Loaded Q-table is not a valid 2D NumPy array. Please check training output.")

# Action mapping
action_labels = {
    0: "Diet",
    1: "Medication",
    2: "Exercise"
}

action_descriptions = {
    0: "Balanced maintained Food Diet will be Better",
    1: "Severe condition Medication is Needed",
    2: "Normal condition Exercise will be Good"
}

def recommend_treatment(user_input):
    try:
        # Reshape and scale input
        user_input = np.array(user_input).reshape(1, -1)
        scaled = scaler.transform(user_input)

        # Predict state
        state = kmeans.predict(scaled)[0]

        # Validate state
        if state >= q_table.shape[0]:
            return "State not found in Q-table. Retrain with more clusters or adjust model."

        # Select best action
        best_action_index = int(np.argmax(q_table[state]))

        # Construct recommendation
        action = action_labels.get(best_action_index, "Unknown")
        description = action_descriptions.get(best_action_index, "No recommendation available.")
        return f"{action}: {description}"

    except Exception as e:
        return "Error occurred during prediction"
