# Task13 -  API  and UI 

from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('features_exc.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index4.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['record']
        if not text_input.strip():
            return render_template('index4.html', prediction="Please enter some text.")

        # Transform and predict
        input_vector = vectorizer.transform([text_input])
        prediction = model.predict(input_vector)[0]
        
        return render_template('index4.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
