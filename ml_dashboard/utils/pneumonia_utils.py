import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model once when the file is imported
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "pneumonia_cnn_model.keras")
model_path = os.path.abspath(model_path)

model = tf.keras.models.load_model(model_path)

def pneumonia_prediction(img_path):
    """
    This function receives the image path, processes the image, and uses the 
    loaded model to predict whether the X-ray image indicates Pneumonia or Normal.
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))  # Same as input size during training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Normalize the image
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    
    # Return prediction (0 = Normal, 1 = Pneumonia)
    if prediction > 0.5:
            return "Prediction: Normal"
    else:
            return "Prediction: Pneumonia"

