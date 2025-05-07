# Task 25 --- Time Series Forecasting model (Auto-ARIMA , Seasonal=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load data
df = pd.read_csv(r"C:\Users\galli\Documents\Office-Tasks\HDHI Admission data.csv")

# Parse dates
df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')
df = df.dropna(subset=['D.O.A'])# if any had missing values will be droped

# Group by month instead of day
monthly_admissions = df.groupby(pd.Grouper(key='D.O.A', freq='M')).size().reset_index(name='Admissions')
monthly_admissions = monthly_admissions.set_index('D.O.A')

print("\nMonthly Admissions:\n", monthly_admissions)

# Split into train/test (last 3 months as test)
train = monthly_admissions.iloc[:-3]
test = monthly_admissions.iloc[-3:]

# Train SARIMA using auto_arima
model = auto_arima(train['Admissions'], seasonal=True, m=12, trace=True,
                   error_action='ignore', suppress_warnings=True, stepwise=True)

# Forecast next 3 months
n_months = 3
forecast = model.predict(n_periods=n_months)

# Plot to visualize forecast
plt.figure(figsize=(12, 6))
# plt.plot(train.index, train['Admissions'], label='Train')
plt.plot(test.index, test['Admissions'], label='Test', color='blue')
#plt.plot(monthly_admissions.index, monthly_admissions['Admissions'], label="Historical Admissions")
#plt.plot(forecast_index, forecast, label='6-Month Forecast', color='red', linestyle='--', marker='o')
plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='--', marker='o')
plt.title('Monthly Admission Forecast for Next 3 Months')
plt.xlabel('Month')
plt.ylabel('Admissions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy
actual = test['Admissions'].values
mae = mean_absolute_error(actual, forecast)
mse = mean_squared_error(actual, forecast)
rmse = np.sqrt(mse)

print(f"\nMAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# Compare forecast vs actual
results_df = pd.DataFrame({
    'Month': test.index.strftime('%Y-%m'),
    'Actual': actual,
    'Predicted': forecast.astype(int)
})
print("\nActual vs Forecasted Admissions (Monthly):\n", results_df)


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
# print("Missing values:\n", missing_values)

# # Convert date column
# df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')
# df = df.dropna(subset=['D.O.A'])

# # Group by date and fill missing dates with 0 admissions
# daily_admissions = df.groupby('D.O.A').size().reset_index(name='Admissions')
# daily_admissions = daily_admissions.set_index('D.O.A').asfreq('D', fill_value=0)

# print(daily_admissions.head())

# # Train-test split (last 30 days for test)
# train = daily_admissions.iloc[:-30]
# test = daily_admissions.iloc[-30:]

# # Train SARIMA model using auto_arima with seasonality
# model = auto_arima(train['Admissions'],
#                    seasonal=True,
#                    m=7,  # weekly seasonality (adjust if needed)
#                    trace=True,
#                    suppress_warnings=True,
#                    stepwise=True,
#                    error_action="ignore")

# # Forecast next 30 days
# forecast = model.predict(n_periods=30)

# # Plot actual and forecast to Visualize the forecast
# plt.figure(figsize=(12, 6))
# plt.plot(train.index, train['Admissions'], label="Train")
# plt.plot(test.index, test['Admissions'], label="Test")
# plt.plot(test.index, forecast, label="Forecast", color='red')
# plt.legend()
# plt.title("Admission Forecast using SARIMA")
# plt.xlabel("Date")
# plt.ylabel("Number of Admissions")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Accuracy metrics
# actual = test['Admissions'].values
# mae = mean_absolute_error(actual, forecast)
# mse = mean_squared_error(actual, forecast)
# rmse = np.sqrt(mse)

# print(f"\nMAE: {mae:.2f}")
# print(f"MSE: {mse:.2f}")
# print(f"RMSE: {rmse:.2f}")

# # Show actual vs predicted
# results_df = pd.DataFrame({
#     'Actual': actual,
#     'Predicted': forecast
# }, index=test.index)

# print("\nActual vs Forecasted Admissions:\n", results_df.head(20))
