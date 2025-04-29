import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load the pre-trained model and vectorizer (replace with your actual model and vectorizer file names)
model = joblib.load('sentiment_model.pkl')  # Replace with your model path
vectorizer = joblib.load('vectorizer.pkl')  # Replace with your vectorizer path

# Function to classify the sentiment of feedback
def classify_feedback(feedback):
    feedback_vector = vectorizer.transform([feedback])
    prediction = model.predict(feedback_vector)
    return prediction[0]

# Streamlit UI
st.title("Hospital Feedback Sentiment Analysis")
st.write("Enter hospital feedback and get the sentiment analysis (Positive, Negative, or Neutral).")

# Input field for user feedback
feedback_input = st.text_area("Enter your feedback:")

# Button to submit feedback and classify sentiment
if st.button("Classify Sentiment"):
    if feedback_input:
        sentiment = classify_feedback(feedback_input)
        if sentiment == 'positive':
            st.success(f"Sentiment: {sentiment.capitalize()}")
        elif sentiment == 'negative':
            st.error(f"Sentiment: {sentiment.capitalize()}")
        else:
            st.warning(f"Sentiment: {sentiment.capitalize()}")
    else:
        st.warning("Please enter some feedback to classify.")
