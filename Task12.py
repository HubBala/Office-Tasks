# Task 12 --- NLP techniques to extract medical terms

import spacy

# Load the scispaCy model
nlp = spacy.load("en_core_sci_sm")  # en_core is a model for medical language

# Sample doctor's note
text = """
The 65-year-old male patient was diagnosed with Type 2 Diabetes Mellitus and Hypertension.
He is currently prescribed metformin and lisinopril. The patient complained of occasional chest pain.
"""

# Process the text
doc = nlp(text)

# Extract and print the named entities
print("Medical Entities Found:\n")
for ent in doc.ents:
    print(f"• {ent.text} [{ent.label_}]")
