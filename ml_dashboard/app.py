# app.py
from flask import Flask, render_template, request
from utils.diagnostic_utils import make_prediction
from utils.Extract_utils import extract_medical_terms
from utils.heart_utils import heart_prediction
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

#--- Api for the Diagnosis of heart disease risk --->
@app.route('/diagnostic', methods=['GET', 'POST'])
def diagnostic():
    prediction_text = ""
    if request.method == 'POST':
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
            float(request.form['thalassemia']),
        ]
        prediction_text = make_prediction(features)

    return render_template('diagnostic.html', prediction_text=prediction_text)

#--- Api for Extraction of medical terms from Doctors notes --->
@app.route('/extract_medical_terms', methods=['GET', 'POST'])
def extract_terms():
    if request.method == 'POST':
        # Get the doctor's note from the form
        text = request.form.get('doctor_note')
        
        # Extract medical terms using the NLP function
        medical_entities = extract_medical_terms(text)
        
        # Render the results on a template
        return render_template('extract_medical_terms.html', medical_entities=medical_entities)
    return render_template('extract_medical_terms.html')

#--- api of Herat_disease Progression --->
model= joblib.load('models/heart_disease_model.pkl')
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    prediction_text = ''
    if request.method == 'POST':
        try:
            # Extracting input values from the form
            features = [
                float(request.form['age']),
                int(request.form['sex']),
                int(request.form['chest_pain_type']),
                float(request.form['resting_blood_pressure']),
                float(request.form['cholesterol']),
                int(request.form['fasting_blood_sugar']),
                int(request.form['rest_ecg']),
                float(request.form['Max_heart_rate']),
                int(request.form['exercise_induced_angina']),
                float(request.form['oldpeak']),
                int(request.form['slope']),
                int(request.form['vessels_colored_by_flourosopy']),
                int(request.form['thalassemia'])
            ]
            input_data = np.array([features])

            prediction = model.predict(input_data)[0]
            prediction_text = f"Predicted Heart Disease Risk Score: {round(prediction, 2)}"
        
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

 

    return render_template('heart_disease.html', prediction_text=prediction_text)


if __name__ == '__main__':
    app.run(debug=True)
