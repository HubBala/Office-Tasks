import pandas as pd
df = pd.read_csv('HeartDiseaseTrain-Test.csv')


df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["fasting_blood_sugar"]= df["fasting_blood_sugar"].map({"Lower than 120 mg/ml": 0, "Greater than 120 mg/ml": 1 })
df["exercise_induced_angina"] = df["exercise_induced_angina"].map({"Yes": 1, "No": 0})
#   Mapping binary categorical values with numerical(1,0)

df["chest_pain_type"] = df["chest_pain_type"].astype("category").cat.codes
df["rest_ecg"] = df["rest_ecg"].astype("category").cat.codes
df["slope"] = df["slope"].astype("category").cat.codes
df["thalassemia"] = df["thalassemia"].astype("category").cat.codes
df["vessels_colored_by_flourosopy"] = df["vessels_colored_by_flourosopy"].astype("category").cat.codes
# Encoding categorical variables 

df.info(), df.head()

#df.to_csv("cleaned HeartDiseaseTrain-Test.csv")

print(" The Dataset is ready work with models ")















































