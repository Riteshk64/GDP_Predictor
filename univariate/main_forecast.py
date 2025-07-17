import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.linear_model import linear_forecast
from models.moving_average_model import moving_average_forecast
from models.exponential_smoothing_model import exponential_smoothing_forecast
from models.arima_model import arima_forecast

import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('univariate/data/India_GDP_Data.csv')
df = data.sort_values('Year').reset_index(drop=True)

years = np.array(df['Year']).reshape(-1, 1)
gdp = np.array(df['GDP_In_Billion_USD'])
forecast_years = np.array([2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])

all_forecasts = {}

# Linear ModelS
all_forecasts['Linear'] = linear_forecast(years, gdp, forecast_years)

# Moving Average Model
all_forecasts['Moving Average'] = moving_average_forecast(gdp, forecast_years)

# Exponential Smoothing Model
all_forecasts['Exponential Smoothing'] = exponential_smoothing_forecast(gdp, forecast_years)

# ARIMA Model
all_forecasts['ARIMA'] = arima_forecast(gdp, forecast_years)

# Comparison of all models
print('\n' + '=' * 60)
print('Comparison of all 4 models')
print('=' * 60)

comparison_df = pd.DataFrame({
    'Year':forecast_years,
    'Linear':[f'${x:.0f}B' for x in all_forecasts['Linear']],
    'Moving Average':[f'${x:.0f}B' for x in all_forecasts['Moving Average']],
    'Exponential Smoothing':[f'${x:.0f}B' for x in all_forecasts['Exponential Smoothing']],
    'ARIMA':[f'${x:.0f}B' for x in all_forecasts['ARIMA']]
})

print(comparison_df.to_string(index=False))

# Visualization
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.plot(df['Year'],df['GDP_In_Billion_USD'],'o-',label='Historical',linewidth=2,markersize=6)

colors=['red','green','orange','purple']
models=['Linear','Moving Average','Exponential Smoothing','ARIMA']

for i, (model, color) in enumerate(zip(models, colors)):
    plt.plot(forecast_years, all_forecasts[model], 'o-',label=model,color=color,alpha=0.7)

plt.title('All Models Comparison')
plt.xlabel('Year')
plt.ylabel('GDP (Billion USD)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2,2,2)
for i, (model, color) in enumerate(zip(models, colors)):
    plt.plot(forecast_years, all_forecasts[model], 'o-',label=model,color=color,linewidth=2)

plt.title('Forecast Comparision (2022-2026)')
plt.xlabel('Year')
plt.ylabel('GDP (Billion USD)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2,2,3)
model_names=list(all_forecasts.keys())
forecast_2026=[all_forecasts[model][-1] for model in models]

bars = plt.bar(model_names,forecast_2026,color=colors,alpha=0.7)
plt.title('2026 GDP Predictions by Model')
plt.ylabel('GDP (Billion USD)')
plt.xticks(rotation=45)

for bar, value in zip(bars, forecast_2026):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,f'${value:.0f}B', ha='center',va='bottom')
plt.tight_layout()
plt.show()

# Exporting the results
results_df = pd.DataFrame({
    'Year':forecast_years,
    'Linear':all_forecasts['Linear'],
    'Moving Average':all_forecasts['Moving Average'],
    'Exponential Smoothing':all_forecasts['Exponential Smoothing'],
    'ARIMA':all_forecasts['ARIMA']
})

results_df.to_csv('univariate/data/gdp_forecast_results.csv', index=False)
comparison_df.to_csv('univariate/data/gdp_forecast_comparison.csv', index=False)