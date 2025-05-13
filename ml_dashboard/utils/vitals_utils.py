import numpy as np
import joblib
from keras.models import load_model

# Load the trained model and scaler once
model = load_model("models/vitals_model.keras")
scaler = joblib.load("models/vitals_scaler.pkl")

def vitals_prediction(age, gender, bp, hr, height=None, weight=None):
    # Example preprocessing: convert gender to numeric
    gender_numeric = 1 if gender.lower() == 'male' else 0

    # You may want to include height and weight if they were part of the training features
    if height is None or weight is None:
        height = 0  # Default or can be left out if you don't want to use them
        weight = 0   # Default or can be left out if you don't want to use them

    # Prepare the input array: Include age, gender, bp, hr, height, and weight
    input_data = np.array([[age, gender_numeric, bp, hr, height, weight]])

    # Reshape the input data to match the expected shape: (1, 3, 6)
    # Assuming you are giving only 1 time step, you repeat it for 3 time steps for simplicity
    input_data_reshaped = np.repeat(input_data, 3, axis=0).reshape(1, 3, 6)
    
    # Scale the input data using the same scaler as used during model training
    input_scaled = scaler.transform(input_data_reshaped.reshape(1, -1)).reshape(1, 3, 6)

    # Make a prediction with the model
    prediction = model.predict(input_scaled)

    return prediction[0][0] if hasattr(prediction[0], '__getitem__') else prediction[0]
