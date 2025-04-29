
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pmdarima import auto_arima
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# # Load data
# df = pd.read_csv(r"C:\Users\galli\Documents\Office-Tasks\HDHI Admission data.csv")
# missing_values = df.isnull().sum()
# print("missing values\n", missing_values)


# df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')
# df = df.dropna(subset=['D.O.A'])

# # Group by date
# daily_admissions = df.groupby('D.O.A').size().reset_index(name='Admissions') # to have the numbert of admissions per day 
# daily_admissions = daily_admissions.set_index('D.O.A').asfreq('D', fill_value=0)# as the dataset has the data that has not in series so making the no admossion days with '0'

# print(daily_admissions)


# # Split into train/test (last 30 days as test)
# train = daily_admissions.iloc[:-30]
# test = daily_admissions.iloc[-30:]

# # Train model
# model = auto_arima(train['Admissions'], seasonal=False, trace=True)
# forecast = model.predict(n_periods=30)

# # Plot
# plt.figure(figsize=(12, 6))
# #plot(train.index, train['Admissions'], label="Train")
# #plt.plot(test.index, test['Admissions'], label="Test")
# plt.plot(test.index, forecast, label="Forecast", color='red')
# plt.legend()
# plt.title("Admission Forecast")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Accuracy
# actual = test['Admissions'].values
# mae = mean_absolute_error(test['Admissions'], forecast)
# mse = mean_squared_error(test['Admissions'], forecast)
# rmse = np.sqrt(mse)

# print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# print("Columns in test DataFrame:", test.columns)

# results_df = pd.DataFrame({
#     'Actual': actual,
#     'Predicted': forecast
# })
# print("\nActual vs Forecasted Admissions:\n", results_df)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load data
df = pd.read_csv(r"C:\Users\galli\Documents\Office-Tasks\HDHI Admission data.csv")
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Convert date column
df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')
df = df.dropna(subset=['D.O.A'])

# Group by date and fill missing dates with 0 admissions
daily_admissions = df.groupby('D.O.A').size().reset_index(name='Admissions')
daily_admissions = daily_admissions.set_index('D.O.A').asfreq('D', fill_value=0)

print(daily_admissions.head())

# Train-test split (last 30 days for test)
train = daily_admissions.iloc[:-30]
test = daily_admissions.iloc[-30:]

# Train SARIMA model using auto_arima with seasonality
model = auto_arima(train['Admissions'],
                   seasonal=True,
                   m=7,  # weekly seasonality (adjust if needed)
                   trace=True,
                   suppress_warnings=True,
                   stepwise=True,
                   error_action="ignore")

# Forecast next 30 days
forecast = model.predict(n_periods=30)

# Plot actual and forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Admissions'], label="Train")
plt.plot(test.index, test['Admissions'], label="Test")
plt.plot(test.index, forecast, label="Forecast", color='red')
plt.legend()
plt.title("Admission Forecast using SARIMA")
plt.xlabel("Date")
plt.ylabel("Number of Admissions")
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy metrics
actual = test['Admissions'].values
mae = mean_absolute_error(actual, forecast)
mse = mean_squared_error(actual, forecast)
rmse = np.sqrt(mse)

print(f"\nMAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# Show actual vs predicted
results_df = pd.DataFrame({
    'Actual': actual,
    'Predicted': forecast
}, index=test.index)

print("\nActual vs Forecasted Admissions:\n", results_df.head(20))
