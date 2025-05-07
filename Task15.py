# Optimized Task 15 --- SVM model for disease progression prediction

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")

# Preprocessing categorical variables
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["fasting_blood_sugar"] = df["fasting_blood_sugar"].map({"Lower than 120 mg/ml": 0, "Greater than 120 mg/ml": 1})
df["exercise_induced_angina"] = df["exercise_induced_angina"].map({"Yes": 1, "No": 0})

# Encode categorical text features
categorical_cols = ["chest_pain_type", "rest_ecg", "slope", "thalassemia", "vessels_colored_by_flourosopy"]
for col in categorical_cols:
    df[col] = df[col].astype("category").cat.codes

# Features and target
X = df.drop(columns=['target'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline with scaling
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Required for SVR performance
    ('svr', SVR())
])

# Grid search with optimized parameters
param_grid = {
    'svr__C': [1, 10],
    'svr__gamma': ['scale', 0.1],
    'svr__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=3,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

# Fit model
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save optimized model
joblib.dump(best_model, 'svm_pipeline_model.pkl')
