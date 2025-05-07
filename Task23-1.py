# Task23-1.py  using the NN for the Recommendation system and a hybrid model content based model
import json
import pandas as pd
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load dataset
with open("datasetB_sample.json", "r") as f:
    data = json.load(f)

# Mapping for condition IDs to names
condition_dict = {cond['id']: cond['name'] for cond in data['Conditions']}

# Build dataset
records = []
for patient in data["Patients"]:
    pid = patient["id"]
    age = patient["age"]
    gender = patient["gender"]
    blood_group = patient["blood_group"]
    condition_count = len(patient["conditions"])
    
    for trial in patient.get("trials", []):
        therapy_id = trial["therapy"]
        success = trial["successful"]
        records.append([age, gender, blood_group, therapy_id, condition_count, success])

df = pd.DataFrame(records, columns=["age", "gender", "blood_group", "therapy_id", "condition_count", "success"])

# Encode categorical variables
le_gender = LabelEncoder()
le_blood = LabelEncoder()
le_therapy = LabelEncoder()

df["gender_enc"] = le_gender.fit_transform(df["gender"])
df["blood_group_enc"] = le_blood.fit_transform(df["blood_group"])
df["therapy_cat"] = le_therapy.fit_transform(df["therapy_id"])

# Features and target
X = df[["age", "gender_enc", "blood_group_enc", "therapy_cat", "condition_count"]].values
y = df["success"].values

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Neural Network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Regression (predict success score)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

# Train
model.fit(X_scaled, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Save model and preprocessors
model.save("hybrid_nn_model.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump({
        "gender": le_gender,
        "blood_group": le_blood,
        "therapy": le_therapy
    }, f)

print("✅ Model and encoders saved successfully!")

print(tf.__version__)