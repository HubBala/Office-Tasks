import pandas as pd


file_path = r"C:\Users\galli\Documents\Office-Tasks\HDHI Admission data.csv\HDHI Admission data.csv" 
df = pd.read_csv(file_path)
# df.info()
missing_values = df.isnull().sum()
print("missing values:", missing_values)

# DROP THE BNP column  as it has 8441 missing values
df = df.drop(columns=['BNP'])

# filling the remaining columns with median and mode
df['EF'] = pd.to_numeric(df['EF'], errors='coerce') # for the values cannot be converted to numeric
df['TLC'] = pd.to_numeric(df['TLC'], errors='coerce')

df['EF'] = df['EF'].fillna(df['EF'].median())
df['TLC'] = df['TLC'].fillna(df['TLC'].median())

df['HB'] = df['HB'].fillna(df['HB'].mode()[0])

df['D.O.A'] = pd.to_datetime(df['D.O.A'], errors='coerce')  # Convert D.O.A to datetime
df['D.O.D'] = pd.to_datetime(df['D.O.D'], errors='coerce')  # Convert D.O.D to datetime

# Calculate the length of stay
df['length_of_stay'] = (df['D.O.D'] - df['D.O.A']).dt.days

df = pd.get_dummies(df, columns=['GENDER', 'RURAL'], drop_first=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['OUTCOME'] = le.fit_transform(df['OUTCOME'])

from sklearn.preprocessing import StandardScaler


numerical_cols = ['AGE', 'DURATION OF STAY', 'duration of intensive unit stay', 'SMOKING', 'ALCOHOL', 'DM', 
                  'HTN', 'CAD', 'PRIOR CMP', 'CKD', 'RAISED CARDIAC ENZYMES', 'EF', 'SEVERE ANAEMIA', 
                  'ANAEMIA', 'STABLE ANGINA', 'ACS', 'STEMI', 'HEART FAILURE', 'HFREF', 'HFNEF', 
                  'VALVULAR', 'AKI', 'CVA INFRACT', 'AF', 'PSVT', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 
                  'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'PULMONARY EMBOLISM']
numerical_cols = [col for col in numerical_cols if col in df.columns]

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

df['DOA_month'] = df['D.O.A'].dt.month
df['DOA_year'] = df['D.O.A'].dt.year
df['DOA_day_of_week'] = df['D.O.A'].dt.dayofweek

df['month_year'] = df['month year'].apply(pd.to_datetime, format='%b-%y')

# Group by 'month_year' to get monthly admissions count
admissions_per_month = df.groupby('month_year').size().reset_index(name='admissions')

# Now admissions_per_month contains the time-series data for forecasting

import matplotlib.pyplot as plt 
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

train = admissions_per_month[:-12]
test = admissions_per_month[-12:]

model = ARIMA(train['admissions'], order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=12)

# plot 
plt.figure(figsize=(10,6))
plt.plot(admissions_per_month['month_year'], admissions_per_month['admissions'], label='Actual Admissions')
plt.plot(pd.date_range(start=test['month_year'].iloc[0], periods=12, freq='ME'), forecast, label='Forecasted Admissions', color='red')
plt.legend()
plt.title('Patient Admission Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.show()