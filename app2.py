# Task15- API

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load(open('svm_pipeline_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    prediction = model.predict([input_data])[0]
    output = "Positive for Heart Disease progression" if prediction == 1 else "Negative for Heart Disease progression"
    return render_template('index1.html', prediction_text=f'Result: {output}')

if __name__ == "__main__":
    app.run(debug=True)
