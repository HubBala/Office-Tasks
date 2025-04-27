import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")

# Data preprocessing
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["fasting_blood_sugar"] = df["fasting_blood_sugar"].map({"Lower than 120 mg/ml": 0, "Greater than 120 mg/ml": 1})
df["exercise_induced_angina"] = df["exercise_induced_angina"].map({"Yes": 1, "No": 0})
df["chest_pain_type"] = df["chest_pain_type"].astype("category").cat.codes
df["rest_ecg"] = df["rest_ecg"].astype("category").cat.codes
df["slope"] = df["slope"].astype("category").cat.codes
df["thalassemia"] = df["thalassemia"].astype("category").cat.codes
df["vessels_colored_by_flourosopy"] = df["vessels_colored_by_flourosopy"].astype("category").cat.codes

# Features and target
X = df.drop(columns=['target'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use smaller subset if needed (comment this if full training is fine)
X_small = X_train[:300]
y_small = y_train[:300]

# Define model and parameter grid
svm = SVR()
param_grid = {
    'C': [1, 10],
    'gamma': ['scale', 0.1],
    'kernel': ['rbf', 'linear']
}

# Grid search with 3-fold CV and parallel processing
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_small, y_small)

# Best model
best_model = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save model
joblib.dump(best_model, 'svm_model.pkl')