# Task 12 -- flask app API or the UI

from flask import Flask, render_template, request
import spacy
app = Flask(__name__)

# Load SciSpaCy model
nlp = spacy.load("en_core_sci_sm")

@app.route('/', methods=['GET', 'POST'])
def index():
    medical_terms = []
    input_text = ""
    
    if request.method == 'POST':
        input_text = request.form['medical_text']
        doc = nlp(input_text)
        medical_terms = [(ent.text, ent.label_) for ent in doc.ents]

    return render_template('index3.html', medical_terms=medical_terms, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
