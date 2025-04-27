import pandas as pd 
import numpy as np 

df = pd.read_csv('Lung Cancer Dataset Innv.csv')
#df. info()
#df.head()

df.columns = df.columns.str.lower().str.replace('_', ' ')

df['pulmonary disease'] = df['pulmonary disease'].map({'YES': 1, 'NO': 0})

summary_stats = df.describe()

print(df.head())
print(summary_stats)
print(df.tail(5))
print(df.sample(10))
df.to_csv("cleaned_lung_cancer_data.csv", index=False)
print("THE DATASET IS CLEANED")


