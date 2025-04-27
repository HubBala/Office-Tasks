import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes dataset.csv")
    return df

# Train model and SHAP explainer
@st.cache_resource
def train_model_and_explainer(df):
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=13)
    model = RandomForestClassifier(max_features=2, n_estimators=100, bootstrap=True)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model, X_train)
    return model, explainer, X_train, X

# Load everything
df = load_data()
model, explainer, X_train, X_full = train_model_and_explainer(df)
feature_names = X_full.columns.tolist()

# UI
st.title("Diabetes Prediction & SHAP Visualization")

st.sidebar.header("Enter Patient Data")
user_input = [st.sidebar.number_input(f"{feat}", value=float(df[feat].mean())) for feat in feature_names]
input_array = np.array(user_input).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    prediction_proba = model.predict_proba(input_array)[0][1]
    st.subheader(f"Prediction: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")
    st.write(f"Probability of being diabetic: {prediction_proba:.2f}")

    st.subheader("SHAP Waterfall Plot")
    expected_value = explainer.expected_value
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use the SHAP values for the positive class
        expected_value = explainer.expected_value[1]
        fig_waterfall = shap.plots._waterfall.waterfall_legacy(
        expected_value, shap_values, feature_names=feature_names, show=False
        )
        st.pyplot(fig_waterfall, bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close(fig_waterfall) # Close the figure to prevent potential display issues

    st.subheader("Global Feature Importance (Bar Plot)")
    shap_values_full = explainer.shap_values(X_train)
    shap.summary_plot(shap_values_full, X_train, plot_type="bar", show=False)
    fig_bar = plt.gcf()
    st.pyplot(fig_bar, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close(fig_bar) # Close the figure

    st.subheader("SHAP Value Summary Table")
    shap_values_full_array = np.array(shap_values_full)
    mean_abs_shap = np.abs(shap_values_full_array).mean(axis=0)
    shap_df = pd.DataFrame(mean_abs_shap, index=X_train.columns, columns=["mean_abs_shap"])
    shap_df = shap_df.sort_values(by="mean_abs_shap", ascending=False)
    st.dataframe(shap_df)

st.write("\n---\nBuilt using Streamlit & SHAP")