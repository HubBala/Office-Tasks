import spacy

# Load the scispaCy model
nlp = spacy.load("en_core_sci_sm")

# Function to extract medical terms
def extract_medical_terms(text):
    doc = nlp(text)
    medical_entities = []
    for ent in doc.ents:
        medical_entities.append(f"{ent.text} [{ent.label_}]")
    return medical_entities
