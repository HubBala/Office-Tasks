# Task 25 --- Time Series Forecasting model (Auto-ARIMA, Seasonal=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load and preprocess data
df = pd.read_csv(r"C:\Users\galli\Documents\Office-Tasks\HDHI Admission data.csv")

# Ensure proper datetime parsing
df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')
df.dropna(subset=['D.O.A'], inplace=True)

# Monthly grouping
monthly_admissions = df.groupby(pd.Grouper(key='D.O.A', freq='M')).size().reset_index(name='Admissions')
monthly_admissions.set_index('D.O.A', inplace=True)

print("\nMonthly Admissions:\n", monthly_admissions)

# Split into train and test (last 3 months as test)
train = monthly_admissions.iloc[:-3]
test = monthly_admissions.iloc[-3:]

# Auto ARIMA to find best SARIMA order
stepwise_model = auto_arima(
    train,
    start_p=0, start_q=0,
    max_p=3, max_q=3,
    seasonal=True, m=12,
    d=None, D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print("\nBest ARIMA model order:", stepwise_model.order, "Seasonal order:", stepwise_model.seasonal_order)

# Fit SARIMAX model using best parameters from auto_arima
sarimax_model = SARIMAX(
    train,
    order=stepwise_model.order,
    seasonal_order=stepwise_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = sarimax_model.fit()

# Forecast next 3 months
n_months = 3
forecast = results.get_forecast(steps=n_months)
forecast_values = forecast.predicted_mean
forecast_index = test.index

# Accuracy metrics
actual = test['Admissions'].values
mae = mean_absolute_error(actual, forecast_values)
mse = mean_squared_error(actual, forecast_values)
rmse = np.sqrt(mse)

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(monthly_admissions.index, monthly_admissions['Admissions'], label="Historical Admissions")
plt.plot(test.index, test['Admissions'], label='Test (Actual)', color='blue')
plt.plot(forecast_index, forecast_values, label='Forecast', color='red', linestyle='--', marker='o')
plt.title('Monthly Admission Forecast for Next 3 Months')
plt.xlabel('Month')
plt.ylabel('Admissions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compare forecast vs actual
results_df = pd.DataFrame({
    'Month': forecast_index.strftime('%Y-%m'),
    'Actual': actual,
    'Predicted': forecast_values.astype(int)
})
print(f"\nMAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")
print("\nActual vs Forecasted Admissions (Monthly):\n", results_df)

# Save to CSV
results_df.to_csv("Monthly_Admissions_Forecast.csv", index=False)

# Model Summary
print("\nModel Summary:\n", results.summary())
