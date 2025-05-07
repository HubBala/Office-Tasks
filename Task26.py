# Task 26 -- Sentiment analysis used Logistic regression with NLP

import pandas as pd
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv('hospital_feedback.csv', encoding='ISO-8859-1')

data['label'] = data['Rating']
data['label'] = data['label'].str.strip()

# To print the count of popsitive, negative, neutral.
print(data['label'].value_counts())

# for converting text to numerical
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['Feedback'])
y = data['label']

# saving the vectorizer to use in the Streamlit code
joblib.dump(vectorizer, "vectorizer.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred, zero_division = 1))

# Saving the model use in the Streamlit code
joblib.dump(model, "sentiment_model.pkl")

def classify_feedback(Feedback):
    feedback_vector = vectorizer.transform([Feedback])
    prediction = model.predict(feedback_vector)
    return prediction[0]

# Test with a new feedback
new_feedback = "The staff is not bad."
print(f"Sentiment: {classify_feedback(new_feedback)}")



# data = pd.read_csv('HealthCare Data.csv', encoding='ISO-8859-1')
# data.info()
# print(data.head())

# unique_categories = data['Patient_Category'].unique()
# print(unique_categories)