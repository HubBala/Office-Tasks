# TASK 17 -- RNN FOR PATIENT VITALS

import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
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

# Train-test split
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Model definition
model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dense(y.shape[1])  # Predict all vitals
])

model.compile(optimizer='adam', loss=MeanSquaredError())

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=157,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Predict on test data
predicted = model.predict(X_test)
predicted_inverse = scaler.inverse_transform(predicted)
actual_inverse = scaler.inverse_transform(y_test)

# Save model and scaler
model.save('Vitals_model.keras')
joblib.dump(scaler, "Vitals_scaler.pkl")

# Visualization: Actual vs Predicted for heart rate (column index 0)
plt.figure(figsize=(10, 5))
plt.plot(actual_inverse[:, 0], label="Actual Heart Rate")
plt.plot(predicted_inverse[:, 0], '--', label="Predicted Heart Rate")
plt.title("Heart Rate Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Heart Rate")
plt.legend()
plt.grid(True)
plt.show()






# # TASK 17 -- RNN FOR PATIENT VITALS 

# import numpy as np  
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import SimpleRNN, Dense
# from tensorflow.keras.losses import MeanSquaredError
# import joblib 

# # Load the data
# data = pd.read_csv("Patient_vitals.csv")

# # Drop the timestamp column
# data = data.drop(columns=['timestamp'])

# # Scale the data
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data)

# # Convert to sequences (use 3 steps to predict next step)
# X, y = [], []
# timesteps = 3
# for i in range(len(scaled_data) - timesteps):
#     X.append(scaled_data[i:i+timesteps])
#     y.append(scaled_data[i+timesteps])

# X = np.array(X)
# y = np.array(y)

# # model
# model = Sequential([
#     SimpleRNN(64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
#     Dense(y.shape[1])  # Predict all vitals
# ])

# model.compile(optimizer='adam', loss=MeanSquaredError())
# model.fit(X, y, epochs=157, verbose=1)


# # Predict on training data
# predicted = model.predict(X)
# predicted_inverse = scaler.inverse_transform(predicted)
# actual_inverse = scaler.inverse_transform(y)

# # model saving for streamlit app
# model.save('Vitals_model.keras')

# # print(model.input_shape)
# joblib.dump(scaler, "scaler.save")


# # Visualization 

# # Plot actual vs predicted for heart rate
# plt.figure(figsize=(10, 5))
# plt.plot(actual_inverse[:, 0], label="Actual Heart Rate")
# plt.plot(predicted_inverse[:, 0], '--', label="Predicted Heart Rate")
# plt.title("Heart Rate Prediction")
# plt.xlabel("Time Steps")
# plt.ylabel("Heart Rate")
# plt.legend()
# plt.grid(True)
# plt.show() 


# The new model 
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler

# # 1. Create synthetic dataset: simulate heart rate data over 2000 minutes
# np.random.seed(42)
# time_steps = 2000
# base_heart_rate = 70 + 5 * np.sin(np.linspace(0, 50, time_steps))  # base sinusoidal pattern
# noise = np.random.normal(0, 1, time_steps)  # add noise
# heart_rate = base_heart_rate + noise

# # Plot example of synthetic heart rate data
# plt.figure(figsize=(12, 3))
# plt.plot(heart_rate[:300])
# plt.title("Synthetic Heart Rate Data (First 300 minutes)")
# plt.xlabel("Time (minutes)")
# plt.ylabel("Heart Rate (bpm)")
# plt.show()

# # 2. Preprocess data: scale between 0 and 1
# scaler = MinMaxScaler(feature_range=(0, 1))
# heart_rate_scaled = scaler.fit_transform(heart_rate.reshape(-1, 1))

# # 3. Create sequences: use past 10 minutes to predict next minute
# def create_sequences(data, seq_length=10):
#     X = []
#     y = []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:i+seq_length])
#         y.append(data[i+seq_length])
#     return np.array(X), np.array(y)

# seq_length = 10
# X, y = create_sequences(heart_rate_scaled, seq_length)

# # 4. Split into train and test sets (80% train, 20% test)
# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# # 5. Build the LSTM model
# model = Sequential()
# model.add(LSTM(50, activation='tanh', input_shape=(seq_length, 1)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

# # 6. Train the model
# history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# # 7. Make predictions
# y_pred_train = model.predict(X_train)
# y_pred_test = model.predict(X_test)

# # Inverse scale back to original heart rate values
# y_train_true = scaler.inverse_transform(y_train)
# y_pred_train_true = scaler.inverse_transform(y_pred_train)
# y_test_true = scaler.inverse_transform(y_test)
# y_pred_test_true = scaler.inverse_transform(y_pred_test)

# # 8. Visualize predictions on test set
# plt.figure(figsize=(12, 5))
# plt.plot(y_test_true, label="True Heart Rate")
# plt.plot(y_pred_test_true, label="Predicted Heart Rate")
# plt.title("Heart Rate Prediction on Test Data")
# plt.xlabel("Time Step")
# plt.ylabel("Heart Rate (bpm)")
# plt.legend()
# plt.show()
