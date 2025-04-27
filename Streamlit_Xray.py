# TASK18 --Streamlit API and UI

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
model = tf.keras.models.load_model("pneumonia_cnn_model.keras")

st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload your image for prediction")

# File uploader
uploaded_file = st.file_uploader("X-Ray Image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    st.write("File uploaded", uploaded_file.name)
    try:
        # Open and display the image
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        st.write(f"Image shape before prediction: {img_array.shape}")

        # Make prediction
        prediction = model.predict(img_array)[0][0]
        st.write("Raw Prediction:", prediction)

        # Show result 
        if prediction > 0.5:
            st.success("Prediction: Normal")
        else:
            st.error("Prediction: Pneumonia")

    except Exception as e:
        st.error(f"Error opening image: {e}")