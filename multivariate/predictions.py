import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from models.arimax_model import arimax_forecast
from models.var_model import var_forecast
from models.prophet_multivariate_model import prophet_multivariate_forecast

# Load and prepare data
df = pd.read_csv("multivariate/data/processed/gdp_factor_india.csv")
df = df[df['Year'].between(1991, 2022)].reset_index(drop=True)
df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
df.set_index('Year', inplace=True)
df.index = pd.to_datetime(df.index, format='%Y')
df = df.asfreq('YS')

gdp = df['GDP (current US$)']
exog = df.drop(columns=['GDP (current US$)'])

n_forecast = 9
forecast_years = pd.date_range(start='2022', periods=n_forecast, freq='YS')

all_forecasts = {}

# ARIMAX
all_forecasts['ARIMAX'] = arimax_forecast(gdp, exog, forecast_years)

# VAR
all_forecasts['VAR'] = var_forecast(df, forecast_years)

# Prophet Multivariate
all_forecasts['Prophet'] = prophet_multivariate_forecast("multivariate/data/processed/gdp_factor_india.csv", forecast_periods=n_forecast)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, gdp, 'o-', label='Historical GDP', linewidth=2)

colors = ['blue', 'brown', 'green']
for (model, color) in zip(all_forecasts.keys(), colors):
    plt.plot(forecast_years, all_forecasts[model], 'o--', label=model, color=color, linewidth=2)

plt.title('GDP Forecasts using ARIMAX, VAR, and Prophet')
plt.xlabel('Year')
plt.ylabel('GDP (current US$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
