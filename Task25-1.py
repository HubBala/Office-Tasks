import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Your file path
file_path = r"C:\Users\galli\Documents\Office-Tasks\HDHI Admission data.csv\HDHI Admission data.csv"
data = pd.read_csv(file_path)
# print(df.head())

# Step 1: Convert 'D.O.A' to datetime
data['D.O.A'] = pd.to_datetime(data['D.O.A'], errors='coerce')

# Step 2: Drop rows with invalid dates (if any)
data = data.dropna(subset=['D.O.A'])

# Step 3: Create a new DataFrame with daily admission counts
admission_trends = data.groupby('D.O.A').size().rename('Admissions').reset_index()
admission_trends = admission_trends.set_index('D.O.A').asfreq('D', fill_value=0)


model = ARIMA(admission_trends['Admissions'], order = (1,1,1))
model_fit = model.fit()

print(model_fit.summary())


# Set 'D.O.A' as index for time series modeling
# admission_trends.set_index('D.O.A', inplace=True)

# Display the prepared time series
# print(admission_trends.head())


# Plotting the time series
plt.figure(figsize=(14,6))
plt.plot(admission_trends, marker='o', linestyle='-')
plt.title('Daily Patient Admissions Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Admissions', fontsize=14)
plt.grid(True)
plt.show()


# Building the model 




'''
# Process data and generate admissions per month
df['month_year'] = pd.to_datetime(df['month year'], format='%b-%y')
admissions_per_month = df.groupby('month_year').size().reset_index(name='admissions')

# Ensure the 'month_year' is the index for time series forecasting
admissions_per_month.set_index('month_year', inplace=True)

# admissions_per_month.index = pd.to_datetime(admissions_per_month.index).asfreq('MS')


# Split into train and test
train = admissions_per_month[:-12]
test = admissions_per_month[-12:]

# Build and fit the ARIMA model
model = ARIMA(train['admissions'], order=(5,1,0))
model_fit = model.fit()

# Forecast for the next 12 months
forecast = model_fit.forecast(steps=12)
'''
''' # Plotting
plt.figure(figsize=(10,6))
plt.plot(admissions_per_month.index, admissions_per_month['admissions'], label='Actual Admissions')
plt.plot(pd.date_range(start=test.index[0], periods=12, freq='ME'), forecast, label='Forecasted Admissions', color='red')
plt.legend()
plt.title('Patient Admission Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.show()
'''
