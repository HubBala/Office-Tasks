import pickle
import numpy as np

# Load models
with open("models/scalerRec.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("models/q_table_Rec.pkl", "rb") as f:
    q_table = pickle.load(f)

# Ensure q_table is a 2D NumPy array
if isinstance(q_table, list):
    q_table = np.array(q_table)

if not isinstance(q_table, np.ndarray) or q_table.ndim != 2:
    raise ValueError("Loaded Q-table is not a valid 2D NumPy array. Please check training output.")

print("Q-table shape:", q_table.shape)

actions = ["Diet", "Medication", "Exercise"]

def recommend_treatment(user_input):
    try:
        print("Raw input:", user_input)
        user_input = np.array(user_input).reshape(1, -1)
        print("Reshaped input:", user_input)

        scaled = scaler.transform(user_input)
        print("Scaled input:", scaled)

        state = kmeans.predict(scaled)[0]
        print("Predicted state:", state)

        if state >= q_table.shape[0]:
            print(f"Invalid state {state}: Q-table only has {q_table.shape[0]} states.")
            return "State not found in Q-table. Retrain with more clusters or adjust model."

        best_action_index = np.argmax(q_table[state])
        print("Best action index:", best_action_index)

        return actions[best_action_index]

    except Exception as e:
        print("Prediction error:", e)
        import traceback
        traceback.print_exc()
        return "Error occurred during prediction"
