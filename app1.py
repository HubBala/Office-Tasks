# Task9 - API 
import os
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
