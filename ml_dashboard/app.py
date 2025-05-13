# app.py
from flask import Flask, render_template, request
from utils.diagnostic_utils import make_prediction
from utils.Extract_utils import extract_medical_terms
from utils.heart_utils import heart_prediction
import os 
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', show_navbar=True)

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

    return render_template('diagnostic.html', prediction_text=prediction_text, show_navbar=False)

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
    return render_template('extract_medical_terms.html', show_navbar=False)

#--- api of Herat_disease Progression --->
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    prediction_text = ""
    if request.method == 'POST':
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
            prediction_text = heart_prediction(features)

    return render_template('heart_disease.html', prediction_text=prediction_text, show_navbar=False)


# <---- API for Vitals model ---->
from utils.vitals_utils import vitals_prediction  

@app.route('/vitals', methods=['GET', 'POST'])
def vitals():
    prediction_text = ""
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            gender = request.form['gender']
            bp = float(request.form['bp'])
            heart_rate = float(request.form['heart_rate'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])

            result = vitals_prediction(age, gender, bp, heart_rate, height, weight)
            prediction_text = f"Prediction: {result:.2f}"

        except Exception as e:
            prediction_text = f"Error occurred: {e}"

    return render_template('vitals.html', prediction_text=prediction_text, show_navbar=False)

# <---- API for Image Classification ---->
from utils.pneumonia_utils import pneumonia_prediction

@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    prediction_text = ""
    if request.method == 'POST':
        try:
            # Get image file from the form
            img_file = request.files['xray_image']
            if img_file:
                # Save the file temporarily
                img_path = os.path.join('static', 'temp_image.jpg')
                img_file.save(img_path)

                # Get prediction from utility function
                result = pneumonia_prediction(img_path)
                prediction_text = result
            
        except Exception as e:
            prediction_text = f"Error occurred: {e}"

    return render_template('pneumonia.html', prediction_text=prediction_text, show_navbar=False)


if __name__ == '__main__':
    app.run(debug=True)
