import joblib

# Load the saved model and vectorizer
model = joblib.load("models/Sentiment_model1.pkl")
vectorizer = joblib.load("models/Sentiment_vectorizer.pkl")  # ✅ load it from disk

def classify_feedback(feedback):
    feedback_vector = vectorizer.transform([feedback])
    prediction = model.predict(feedback_vector)
    return prediction[0]
