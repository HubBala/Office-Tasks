# TASK 17 -- RNN FOR PATIENT VITALS 

import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.losses import MeanSquaredError
import joblib 

# Load the data
data = pd.read_csv("Patient_vitals.csv")

# Drop the timestamp column
data = data.drop(columns=['timestamp'])

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Convert to sequences (use 3 steps to predict next step)
X, y = [], []
timesteps = 3
for i in range(len(scaled_data) - timesteps):
    X.append(scaled_data[i:i+timesteps])
    y.append(scaled_data[i+timesteps])

X = np.array(X)
y = np.array(y)

# model
model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dense(y.shape[1])  # Predict all vitals
])

model.compile(optimizer='adam', loss=MeanSquaredError())
model.fit(X, y, epochs=157, verbose=1)


# Predict on training data
predicted = model.predict(X)
predicted_inverse = scaler.inverse_transform(predicted)
actual_inverse = scaler.inverse_transform(y)

# model saving for streamlit app
model.save('Vitals_model.keras')

# print(model.input_shape)
joblib.dump(scaler, "scaler.save")


# Visualization 

# Plot actual vs predicted for heart rate
plt.figure(figsize=(10, 5))
plt.plot(actual_inverse[:, 0], label="Actual Heart Rate")
plt.plot(predicted_inverse[:, 0], '--', label="Predicted Heart Rate")
plt.title("Heart Rate Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Heart Rate")
plt.legend()
plt.grid(True)
plt.show() 