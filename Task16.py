# # Task 16 - 1 is better than this 

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import class_weight
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# import joblib

# # 1. Load Dataset
# df = pd.read_csv('HeartDiseaseTrain-Test.csv')

# # 2. Feature Order
# feature_order = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholestoral',
#                  'fasting_blood_sugar', 'rest_ecg', 'Max_heart_rate',
#                  'exercise_induced_angina', 'oldpeak', 'slope',
#                  'vessels_colored_by_flourosopy', 'thalassemia']

# # 3. Categorical Mapping
# df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
# df["chest_pain_type"] = df["chest_pain_type"].map({"Asymptomatic": 0, "Atypical": 1, "Non-Anginal pain": 2, "Typical Angina": 3})
# df["fasting_blood_sugar"] = df["fasting_blood_sugar"].map({"Lower than 120 mg/ml": 0, "Greater than 120 mg/ml": 1})
# df["rest_ecg"] = df["rest_ecg"].map({"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2})
# df["exercise_induced_angina"] = df["exercise_induced_angina"].map({"Yes": 1, "No": 0})
# df["slope"] = df["slope"].map({"upsloping": 0, "Flat": 1, "downsloping": 2})
# df["vessels_colored_by_flourosopy"] = df["vessels_colored_by_flourosopy"].map({"Zero": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4})
# df["thalassemia"] = df["thalassemia"].map({"normal": 0, "fixed defect": 1, "Reversible defect": 2, "No": 3})

# # 4. Features and Target
# X = df[feature_order]
# y = df['target']

# # 5. Split Data
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # 6. Standardize
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)

# # 7. Class Weights
# cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weights = dict(enumerate(cw))

# # 8. Build Model
# model = Sequential([
#     Dense(128, activation='relu', input_dim=X_train.shape[1]),
#     Dropout(0.2),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # 9. Compile
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # 10. Train
# history = model.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=16,
#     validation_split=0.1,
#     class_weight=class_weights,
#     verbose=1
# )

# # 11. Evaluate
# train_acc = history.history['accuracy'][-1]
# val_acc = history.history['val_accuracy'][-1]

# print(f"\nFinal Training Accuracy: {train_acc:.4f}")
# print(f"Final Validation Accuracy: {val_acc:.4f}")

# # 12. Save Model & Scaler
# model.save('heart_model.keras')
# joblib.dump(scaler, "heart_model_scaler.save")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Step 1: Load dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")

# Step 2: One-hot encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 3: Separate features and target
X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

# Step 4: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Step 6: Compute class weights
cw = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(cw))

# Step 7: Build the neural network model with Dropout
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Step 8: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 9: Train the model with class weights
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    verbose=1,
    callbacks=[early_stopping]
)

# Step 10: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Step 11: Report final training and validation accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# Step 12: Save model and scaler
model.save("heart_disease_model.keras")
joblib.dump(scaler, 'scaler.pkl')
