import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def forecast_exog_arima(exog_df, forecast_years):
    exog_forecast = pd.DataFrame(index=forecast_years, columns=exog_df.columns)
    
    for col in exog_df.columns:
        series = exog_df[col]
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(forecast_years))
        exog_forecast[col] = forecast.values

    return exog_forecast
