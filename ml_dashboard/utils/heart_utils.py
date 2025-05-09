import joblib
import numpy as np 

def load_model():

    model = joblib.load('models/heart_disease_model.pkl')
    return model

def heart_prediction(features):
    """
    Makes a prediction using the loaded model.

    Parameters:
    - features: List or array-like of numerical input features (same order as used in training)

    Returns:
    - Prediction value (float)
    """
    model = load_model()
    
    try:
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return round(float(prediction[0]), 2)
    except Exception as e:
        return f"Prediction error: {str(e)}"