import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df_cleaned = pd.read_csv("cleaned HeartDiseaseTrain-Test.csv")  # cleaned dataset 


# Encoding categorical variables
categorical_cols = ["sex", "chest_pain_type", "fasting_blood_sugar", "rest_ecg", 
                    "exercise_induced_angina", "slope", "vessels_colored_by_flourosopy", "thalassemia"]

encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_cols = encoder.fit_transform(df_cleaned[categorical_cols])
encoded_col_names = encoder.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_cols, columns=encoded_col_names)

# Drop original categorical columns and concatenate encoded ones
df_cleaned = df_cleaned.drop(columns=categorical_cols).reset_index(drop=True)
df_cleaned = pd.concat([df_cleaned, df_encoded], axis=1)

# Standardizing numerical features
scaler = StandardScaler()
num_cols = ["age", "resting_blood_pressure", "cholestoral", "Max_heart_rate", "oldpeak"]
df_cleaned[num_cols] = scaler.fit_transform(df_cleaned[num_cols])

# Splitting dataset
X = df_cleaned.drop(columns=["target"])
y = df_cleaned["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy, classification_rep

print (f"Accuracy:{accuracy:.2f}")
print(f"\nClassification Report:\n", classification_rep)

sample = X_test.iloc[0:1]
prediction = model.predict(sample)

sample = X_test.drop(columns=['Unnamed: 0'], errors='ignore').iloc[0:1]

sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Patient: {sample_dict}")
print(f"Prediction: {'Diseased' if prediction [0] == 1 else 'NotDiseased'}")

