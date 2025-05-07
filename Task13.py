# Task 13 --- Basic Text classification model using NaviBayes model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# data
X_data = [
    "Patient experiences shortness of breath, wheezing, and chest tightness.",
    "Complains of joint stiffness, swelling, and reduced motion.",
    "Reports persistent headache, nausea, and sensitivity to light.",
    "X-ray shows lung inflammation; patient has cough and fever.",
    "Wheezing observed during physical activity; triggered by allergens.",
    "Patient has swollen knees and complains of pain during movement.",
    "Intense throbbing headache on one side, followed by blurred vision.",
    "Coughing up mucus, fever, and difficulty breathing.",
    "Frequent asthma attacks, uses inhaler twice daily.",
    "Severe migraines causing light and sound sensitivity.",
    "Pain in multiple joints, especially in the morning.",
    "Patient shows signs of pneumonia and decreased oxygen levels."
]

y_labels = [
    "Asthma",
    "Arthritis",
    "Migraine",
    "Pneumonia",
    "Asthma",
    "Arthritis",
    "Migraine",
    "Pneumonia",
    "Asthma",
    "Migraine",
    "Arthritis",
    "Pneumonia"
]

df = pd.DataFrame({'record': X_data, 'label': y_labels})

# split
X_train, X_test, y_train, y_test = train_test_split(
    df['record'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Save model and vectorizer by dump

joblib.dump(model, 'features_exc.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')


# # Optimized Task 13 --- Basic Text classification using Naive Bayes

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Data
# X_data = [
#     "Patient experiences shortness of breath, wheezing, and chest tightness.",
#     "Complains of joint stiffness, swelling, and reduced motion.",
#     "Reports persistent headache, nausea, and sensitivity to light.",
#     "X-ray shows lung inflammation; patient has cough and fever.",
#     "Wheezing observed during physical activity; triggered by allergens.",
#     "Patient has swollen knees and complains of pain during movement.",
#     "Intense throbbing headache on one side, followed by blurred vision.",
#     "Coughing up mucus, fever, and difficulty breathing.",
#     "Frequent asthma attacks, uses inhaler twice daily.",
#     "Severe migraines causing light and sound sensitivity.",
#     "Pain in multiple joints, especially in the morning.",
#     "Patient shows signs of pneumonia and decreased oxygen levels."
# ]

# y_labels = [
#     "Asthma", "Arthritis", "Migraine", "Pneumonia",
#     "Asthma", "Arthritis", "Migraine", "Pneumonia",
#     "Asthma", "Migraine", "Arthritis", "Pneumonia"
# ]

# # Create DataFrame
# df = pd.DataFrame({'record': X_data, 'label': y_labels})

# # Split
# X_train, X_test, y_train, y_test = train_test_split(
#     df['record'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
# )

# # Pipeline for optimization and scalability
# model_pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=1)),
#     ('clf', MultinomialNB(alpha=0.5))
# ])

# # Train
# model_pipeline.fit(X_train, y_train)

# # Evaluate
# y_pred = model_pipeline.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# # Save the pipeline
# joblib.dump(model_pipeline, 'text_classification_pipeline.pkl')
