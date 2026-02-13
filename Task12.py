# updated
# Task 12 --- NLP techniques to extract medical terms

import spacy

# Load the scispaCy model
# nlp = spacy.load("en_core_sci_sm")  # en_core is a model for medical language that can specify the different medical terms

# nlp = spacy.load("en_ner_bionlp13cg_md") # many bio terms 

nlp = spacy.load("en_ner_bc5cdr_md") # disease | chemicals 
# Sample doctor's note

#text = """
#The 65-year-old male patient was diagnosed with Type 2 Diabetes Mellitus and Hypertension.
#He is currently prescribed metformin and lisinopril. The patient complained of occasional chest pain.
#""" 

text = """The 72-year-old female patient with a history of Type 2 Diabetes Mellitus, 
Hypertension, and Chronic Kidney Disease was admitted to the hospital 
complaining of severe chest pain, shortness of breath, and dizziness.

Her past medical history includes Coronary Artery Disease, Myocardial Infarction, 
and Hyperlipidemia. She is currently taking metformin, insulin, lisinopril, 
atorvastatin, and aspirin.

On examination, the patient was found to have elevated blood pressure and 
tachycardia. Laboratory tests revealed increased troponin levels and elevated 
creatinine. An ECG suggested acute myocardial infarction.

The patient was started on heparin and nitroglycerin therapy. 
A coronary angiography was performed, and a stent was placed 
in the left anterior descending artery.

Further management included clopidogrel, beta-blockers, and statin therapy. 
The patient was advised lifestyle modifications including diet control 
and regular exercise to manage diabetes and hypertension.
"""
# Process the text
doc = nlp(text)

# Extract and print the named entities
print("Medical Entities Found:\n")
for ent in doc.ents:
    print(f"• {ent.text} [{ent.label_}]")
