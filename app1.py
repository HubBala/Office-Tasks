# Task9 - API 
""" import os
from flask import Flask, render_template, request
import joblib  # To load the trained model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("diagnostic_model.pkl")  # Ensure you have a saved model

@app.route('/') # base root for the url
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) #  the root for index 
def predict():
    try:
        # Extract user inputs from form
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['chest_pain_type']),
            float(request.form['resting_blood_pressure']),
            float(request.form['cholesterol']),
            float(request.form['fasting_blood_sugar']),
            float(request.form['rest_ecg']),
            float(request.form['Max_heart_rate']),
            float(request.form['exercise_induced_angina']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['vessels_colored_by_flourosopy']),
            float(request.form['thalassemia'])
        ]
        
        # Convert to NumPy array for model input
        features_array = np.array([features])
        
        # Make prediction
        prediction = model.predict(features_array)
        result = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"
        
        return render_template('index.html', prediction_text=f'Prediction: {result}') # gives the result to the UI
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

"""
import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")

# ---------------------------------------------------
# Load Model Safely
# ---------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "diagnostic_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ---------------------------------------------------
# Helper function for safe float conversion
# ---------------------------------------------------
def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

# ---------------------------------------------------
# Home Route
# ---------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------------------------------------------
# Prediction Route
# ---------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html',
                               prediction_text="Error: Model not loaded.")

    try:
        # Extract and validate inputs
        features = [
            safe_float(request.form.get('age')),
            safe_float(request.form.get('sex')),
            safe_float(request.form.get('chest_pain_type')),
            safe_float(request.form.get('resting_blood_pressure')),
            safe_float(request.form.get('cholesterol')),
            safe_float(request.form.get('fasting_blood_sugar')),
            safe_float(request.form.get('rest_ecg')),
            safe_float(request.form.get('max_heart_rate')),
            safe_float(request.form.get('exercise_induced_angina')),
            safe_float(request.form.get('oldpeak')),
            safe_float(request.form.get('slope')),
            safe_float(request.form.get('vessels_colored_by_fluoroscopy')),
            safe_float(request.form.get('thalassemia'))
        ]

        # Convert to numpy array
        features_array = np.array([features])

        # Make prediction
        prediction = model.predict(features_array)[0]

        # Confidence (if model supports it)
        try:
            probability = model.predict_proba(features_array)[0][1]
            confidence_text = f" (Confidence: {round(probability * 100, 2)}%)"
        except:
            confidence_text = ""

        result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

        return render_template(
            'index.html',
            prediction_text=f"Prediction: {result}{confidence_text}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error during prediction: {str(e)}"
        )

# ---------------------------------------------------
# Run App
# ---------------------------------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
