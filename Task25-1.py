import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load and preprocess data
df = pd.read_csv(r"C:\Users\galli\Documents\Office-Tasks\HDHI Admission data.csv")
df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')
df = df.dropna(subset=['D.O.A'])

# Group by month
monthly_admissions = df.groupby(pd.Grouper(key='D.O.A', freq='M')).size().reset_index(name='Admissions')
monthly_admissions = monthly_admissions.set_index('D.O.A')

# Rolling trend
monthly_admissions['Rolling_Mean'] = monthly_admissions['Admissions'].rolling(window=3).mean()

print("\nMonthly Admissions:\n", monthly_admissions)

# Train-test split
train = monthly_admissions.iloc[:-3]
test = monthly_admissions.iloc[-3:]

# Train model
model = auto_arima(train['Admissions'], seasonal=True, m=12, trace=True,
                   error_action='ignore', suppress_warnings=True, stepwise=True)

# Forecast next 3 months
n_months = 3
forecast = model.predict(n_periods=n_months)
forecast_index = pd.date_range(start=monthly_admissions.index[-1] + pd.offsets.MonthBegin(1),
                               periods=n_months, freq='M')

# Plot trend and forecast together
plt.figure(figsize=(14, 6))
plt.plot(monthly_admissions.index, monthly_admissions['Admissions'], label='Actual Admissions')
plt.plot(monthly_admissions.index, monthly_admissions['Rolling_Mean'], label='3-Month Trend', color='orange')
plt.plot(forecast_index, forecast, label='Forecast (3 Months)', color='red', linestyle='--', marker='o')
plt.title('Patient Admission Trends with Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Forecast next 6 months
n_months = 6
future_forecast = model.predict(n_periods=n_months)
future_dates = pd.date_range(monthly_admissions.index[-1] + pd.offsets.MonthBegin(1), periods=n_months, freq='M')

# Combine for final full forecast plot
full_plot_df = monthly_admissions[['Admissions']].copy()
forecast_df = pd.DataFrame({'Admissions': future_forecast}, index=future_dates)
combined_df = pd.concat([full_plot_df, forecast_df])

# Plot full admissions + forecast
plt.figure(figsize=(14, 6))
plt.plot(full_plot_df.index, full_plot_df['Admissions'], label='Historical Admissions')
plt.plot(forecast_df.index, forecast_df['Admissions'], label='Forecast (6 Months)', color='green', linestyle='--', marker='o')
plt.title('Hospital Admissions Forecast Including Trends')
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluation
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

# Optional: Save forecast for report
future_df = pd.DataFrame({
    'Month': future_dates,
    'Forecasted_Admissions': future_forecast.astype(int)
})
print("\nForecast for Next 6 Months:\n", future_df)

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(monthly_admissions['Admissions'], model='additive', period=12)
decomposition.plot()
plt.show()
