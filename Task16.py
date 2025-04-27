import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# 1. Load Dataset
df = pd.read_csv('HeartDiseaseTrain-Test.csv')

# 2. Feature Order
feature_order = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholestoral',
                 'fasting_blood_sugar', 'rest_ecg', 'Max_heart_rate',
                 'exercise_induced_angina', 'oldpeak', 'slope',
                 'vessels_colored_by_flourosopy', 'thalassemia']

# 3. Categorical Mapping
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["chest_pain_type"] = df["chest_pain_type"].map({"Asymptomatic": 0, "Atypical": 1, "Non-Anginal pain": 2, "Typical Angina": 3})
df["fasting_blood_sugar"] = df["fasting_blood_sugar"].map({"Lower than 120 mg/ml": 0, "Greater than 120 mg/ml": 1})
df["rest_ecg"] = df["rest_ecg"].map({"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2})
df["exercise_induced_angina"] = df["exercise_induced_angina"].map({"Yes": 1, "No": 0})
df["slope"] = df["slope"].map({"upsloping": 0, "Flat": 1, "downsloping": 2})
df["vessels_colored_by_flourosopy"] = df["vessels_colored_by_flourosopy"].map({"Zero": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4})
df["thalassemia"] = df["thalassemia"].map({"normal": 0, "fixed defect": 1, "Reversible defect": 2, "No": 3})

# 4. Features and Target
X = df[feature_order]
y = df['target']

# 5. Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 6. Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 7. Class Weights
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(cw))

# 8. Build Model
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 9. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 10. Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    class_weight=class_weights,
    verbose=1
)

# 11. Evaluate
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print(f"\nFinal Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")

# 12. Save Model & Scaler
model.save('heart_model.keras')
joblib.dump(scaler, "heart_model_scaler.save")
