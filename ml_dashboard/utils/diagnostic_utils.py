# utils.py
import joblib
import numpy as np

# Load the diagnostic model (adjust the path if needed)
def load_model():
    # with open('models/diagnostic_model.pkl', 'rb') as model_file:
    #     model = pickle.load(model_file)
    # return model
    model = joblib.load('models/diagnostic_model.pkl')
    return model

# Make prediction
def make_prediction(features):
    model = load_model()
    features = np.array(features).reshape(1, -1)  # Reshape for single sample
    prediction = model.predict(features)

    # Return prediction outcome
    return "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"
