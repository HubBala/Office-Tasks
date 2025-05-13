import numpy as np
import joblib
from keras.models import load_model

# Load model and scaler
model = load_model("models/Vitals_model.keras")
scaler = joblib.load("models/Vitals_scaler.pkl")

def vitals_prediction(age, gender, bp, heart_rate, height, weight):
    # Convert gender to 0/1
    gender_numeric = 1 if str(gender).lower() == 'male' else 0

    # Make input array
    input_data = np.array([[age, gender_numeric, bp, heart_rate, height, weight]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Reshape for LSTM input
    input_reshaped = np.repeat(input_scaled, 3, axis=0).reshape(1, 3, 6)

    # Predict
    prediction = model.predict(input_reshaped)

    return prediction[0][0]
