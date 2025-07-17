import numpy as np

def moving_average_forecast(gdp, forecast_years):
    window = 3
    last_values = gdp[-window:]
    base_ma = np.mean(last_values)
    recent_trend = np.mean(np.diff(gdp[-4:]))

    ma_forecast = []
    for i in range(len(forecast_years)):
        forecast_value = base_ma + recent_trend * (i + 1)
        ma_forecast.append(forecast_value)
    return np.array(ma_forecast)
