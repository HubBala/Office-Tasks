import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model =load_model("models/heart_disease_model.keras")  # Use keras.models.load_model() if it’s a .h5
scaler = joblib.load("models/scaler.pkl")

def heart_prediction(user_input):
    # Create DataFrame from input list
    columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol',
               'fasting_blood_sugar', 'rest_ecg', 'Max_heart_rate', 'exercise_induced_angina',
               'oldpeak', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia']

    df_input = pd.DataFrame([user_input], columns=columns)

    # One-hot encode like training
    df_encoded = pd.get_dummies(df_input, drop_first=True)

    # Ensure same columns order and structure as during training
    expected_cols = joblib.load('models/expected_columns.pkl')  # save during training
    for col in expected_cols:
        if col not in df_encoded:
            df_encoded[col] = 0  # Add missing columns with 0
    df_encoded = df_encoded[expected_cols]  # ensure order

    # Scale
    X_scaled = scaler.transform(df_encoded)

    # Predict
    prediction = model.predict(X_scaled)[0][0]
    result = "High Progression" if prediction > 0.5 else "Low Progression"

    return f"Predicted heart disease Progression: {result} (Probability: {prediction:.2f})"
