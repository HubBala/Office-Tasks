import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)

# Load data
file_path = r"C:\Users\galli\Documents\Office-Tasks\HDHI Admission data.csv"
df = pd.read_csv(file_path)

# Ensure datetime conversion
df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')

# Drop rows where date conversion failed
df.dropna(subset=['D.O.A'], inplace=True)

# Extract 'month_year' in YYYY-MM format
df['month_year'] = df['D.O.A'].dt.to_period('M').astype(str)

# Monthly patient count
admissions_per_month = df.groupby('month_year').size().reset_index(name='admissions')
admissions_per_month = admissions_per_month.sort_values('month_year')

# Convert month_year to datetime
admissions_per_month['month_year'] = pd.to_datetime(admissions_per_month['month_year'])

# Split train and test
train = admissions_per_month[:-12]
test = admissions_per_month[-12:]

# ADF Test for stationarity
result = adfuller(train['admissions'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
if result[1] > 0.05:
    print("Series is likely non-stationary. Differencing will be used.")
else:
    print("Series is likely stationary.")

# Auto ARIMA with simpler model and no seasonality
model = auto_arima(train['admissions'],
                   seasonal=False,
                   d=None,              # Let auto_arima decide differencing
                   max_p=2, max_q=2, max_d=2,  # Restrict model complexity
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

# Summary
print(model.summary())

# Diagnostics
model.plot_diagnostics()
plt.tight_layout()
plt.show()

# Forecast next 12 months
n_periods = 12
forecast = model.predict(n_periods=n_periods)

# Create forecast date range
forecast_dates = pd.date_range(start=test['month_year'].iloc[0], periods=n_periods, freq='MS')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(admissions_per_month['month_year'], admissions_per_month['admissions'], label='Actual Admissions')
plt.plot(forecast_dates, forecast, label='Forecasted Admissions', color='red', linestyle='--')
plt.title('Patient Admission Forecast (Reduced Overfitting)')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Evaluation
actual = test['admissions'].values
mae = mean_absolute_error(actual, forecast)
mse = mean_squared_error(actual, forecast)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# Comparison table
comparison = pd.DataFrame({
    'Date': forecast_dates,
    'Actual': actual,
    'Forecast': forecast
})
print("\nForecast vs Actual:")
print(comparison)





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
