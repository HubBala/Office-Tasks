# Task9 - for decision tree and random forest

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt



df = pd.read_csv('HeartDiseaseTrain-Test.csv')
# df.info()
# print(df.head())

df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["fasting_blood_sugar"]= df["fasting_blood_sugar"].map({"Lower than 120 mg/ml": 0, "Greater than 120 mg/ml": 1 })
df["exercise_induced_angina"] = df["exercise_induced_angina"].map({"Yes": 1, "No": 0})

df["chest_pain_type"] = df["chest_pain_type"].astype("category").cat.codes
df["rest_ecg"] = df["rest_ecg"].astype("category").cat.codes
df["slope"] = df["slope"].astype("category").cat.codes
df["thalassemia"] = df["thalassemia"].astype("category").cat.codes
df["vessels_colored_by_flourosopy"] = df["vessels_colored_by_flourosopy"].astype("category").cat.codes

# features and target split 
X = df[['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholestoral', 'fasting_blood_sugar',
        'rest_ecg', 'Max_heart_rate', 'exercise_induced_angina', 'oldpeak', 'slope', 'vessels_colored_by_flourosopy', 
        'thalassemia']]

y = df['target'] 

#df.info()

# train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Random forest 
rf_classifier = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# Decision Tree
dt_classifier = DecisionTreeClassifier(max_depth = 4, random_state = 42)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Results 
print(f"DecisionTree Accuracy:", dt_accuracy)
print(f"RandomForest Accuracy:", rf_accuracy)

# ploting bar graph 

models = ['Decision Tree', 'Random Forest']
accuracies = [0.824, 0.985]  # Convert percentages to decimal

# Bar Plot 
plt.figure(figsize=(6,4))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.ylim(0.8, 1.0)  # limits 
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison of RandomForest vs DecisionTree')

# Show accuracy values on bars
'''for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.2%}", ha='center', fontsize=12, fontweight='bold')
'''
plt.show()

# saving the model random forest
import joblib
joblib.dump(rf_classifier, 'diagnostic_model.pkl')

