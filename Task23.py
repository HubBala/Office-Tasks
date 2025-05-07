# Recommendation System of Therapies for patients by using SVD model

import json 
import pandas as pd 
from surprise import Reader, SVD, Dataset
from surprise.model_selection import train_test_split
import pickle

with open ("datasetB_sample.json", "r") as f:
    data = json.load(f)

print(data.keys())
# View some conditions
print("Conditions Sample:", data['Conditions'][:2])

# View some patients
print("Patients Sample:", data['Patients'][:2])

# View some therapies
print("Therapies Sample:", data['Therapies'][:2])

therapy_id_to_name = {t['id']: t['name'] for t in data['Therapies']}

# extracting the patients trial data 
interaction_data = []

for patient in data['Patients']:
    patient_id = patient['id']
    trials = patient.get('trials', [])

    for trial in trials:
        therapy_id = trial['therapy']
        success = trial['successful']

        if patient_id is not None and therapy_id and success is not None:
            interaction_data.append([patient_id, therapy_id, success])

# converting those into a pandas data frame
df = pd.DataFrame(interaction_data, columns =['patient_id', 'therapy_id', 'success'])

print(df.head())
print(df.info())

# print(f"Total patients: {len(data['Patients'])}")

reader = Reader(rating_scale=(0, 100))
data = Dataset.load_from_df(df[['patient_id', 'therapy_id', 'success']], reader)

# train test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)

# save the model
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(model, f)

df.to_csv('patient_therapy_success.csv', index=False)

# sample outcome of id=0    
example_patient_id = 12
all_therapies = df['therapy_id'].unique()

# Recommend top-N therapies the patient hasn't tried
def get_recommendations(patient_id, n=5):
    tried_therapies = df[df['patient_id'] == patient_id]['therapy_id'].unique()
    untried = [t for t in all_therapies if t not in tried_therapies]
    
    pred_scores = [(therapy_id, model.predict(patient_id, therapy_id).est) for therapy_id in untried]
    sorted_preds = sorted(pred_scores, key=lambda x: x[1], reverse=True)
    
    return sorted_preds[:n]

# Show top 5 therapy recommendations for patient 0
recommended = get_recommendations(12)
print("Top Therapy Recommendations for Patient 12:")
for therapy_id, score in recommended:
    therapy_name = therapy_id_to_name.get(therapy_id, "Unknown Therapy")
    print(f"Therapy: {therapy_name} ({therapy_id}), Predicted Success Score: {score:.2f}")

# This SVD- Singular Value Decomposition model is a 
# pure recommender system based on patient-therapy-success history.
# Just like Netflix as it recommend movies based on movie rating.


