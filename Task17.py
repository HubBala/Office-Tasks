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
