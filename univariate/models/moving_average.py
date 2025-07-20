import numpy as np
from sklearn.linear_model import LinearRegression

def moving_average(years, gdp, forecast_years):
    best_window = 3
    best_error = float('inf')

    for window in [3, 4, 5]:
        if window < len(gdp):
            try:
                last_values = gdp[-window-2:-2]
                predicted = np.mean(last_values)
                actual = np.mean(gdp[-2:])
                error = abs(predicted - actual)
                if error < best_error:
                    best_error = error
                    best_window = window
            except:
                continue

    recent_years = years[-8:]
    recent_gdp = gdp[-8:]
    trend_model = LinearRegression().fit(recent_years.reshape(-1, 1), recent_gdp)
    annual_trend = trend_model.coef_[0]

    base_ma = np.mean(gdp[-best_window:])
    ma_forecast = [base_ma + annual_trend * (i+1) for i in range(len(forecast_years))]

    return np.array(ma_forecast)
