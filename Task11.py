# Task 11 - multi class classification

import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


df = pd.read_csv('heart_disease_uci.csv') # read the dataset

#df.info()

# drop irrelavent columns 
df.drop(columns=["id","dataset"], inplace=True)

#print(df.nunique())

# too many missing values 
df.dropna(subset=["ca", "slope", "thal"], inplace=True)

# filling missing values of in cat col by mean 
for col in ["trestbps", "chol", "thalch", "oldpeak"]:
    df[col] = df[col].fillna(df[col].mean())
    df = df.infer_objects(copy=False)

# these by mode 
for col in ["fbs", "restecg", "exang"]:
    df[col] = df[col].fillna(df[col].mode()[0])
    df = df.infer_objects(copy=False)


# converting objects to numerical
df["sex"] = df["sex"].map({"Male": 1, "Female": 0 })
df["cp"] = df["cp"].map({"typical angina": 0, "asymptomatic": 1, "non-anginal": 2, "atypical angina": 3})
df["fbs"] = df["fbs"].map({"TRUE": 0, "FALSE": 1})
df["restecg"] = df["restecg"].map({"lv hypertropy": 0, "normal": 1, "st-t abnormality": 2})
df["exang"] = df["exang"].map({"TRUE": 0, "FALSE": 1})
df["slope"] = df["slope"].map({"upsloping": 0, "flat": 1, "downsloping": 2})
df["thal"] = df["thal"].map({"fixed defeat":0, "normal": 1, "reversable": 2})

#df.info()

#df.to_csv('data.csv')


# Decission Tree Model for Multi class classification problem

X = df.drop("num", axis=1) # 1 for col, 0 for row
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth= 4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

np.shape(shap_values.values)
print(shap_values.shape)

