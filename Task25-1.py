import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Your file path
file_path = r"C:\Users\galli\Documents\Office-Tasks\HDHI Admission data.csv\HDHI Admission data.csv"
df = pd.read_csv(file_path)

# Process data and generate admissions per month
df['month_year'] = pd.to_datetime(df['month year'], format='%b-%y')
admissions_per_month = df.groupby('month_year').size().reset_index(name='admissions')

# Ensure the 'month_year' is the index for time series forecasting
admissions_per_month.set_index('month_year', inplace=True)

admissions_per_month.index = pd.to_datetime(admissions_per_month.index).asfreq('MS')


# Split into train and test
train = admissions_per_month[:-12]
test = admissions_per_month[-12:]

# Build and fit the ARIMA model
model = ARIMA(train['admissions'], order=(5,1,0))
model_fit = model.fit()

# Forecast for the next 12 months
forecast = model_fit.forecast(steps=12)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(admissions_per_month.index, admissions_per_month['admissions'], label='Actual Admissions')
plt.plot(pd.date_range(start=test.index[0], periods=12, freq='M'), forecast, label='Forecasted Admissions', color='red')
plt.legend()
plt.title('Patient Admission Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.show()
