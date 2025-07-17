import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from multivariate.models.exog_forecast_model import forecast_exog_arima

def arimax_forecast(gdp_series, exog_df, forecast_years):
    arimax_model = SARIMAX(gdp_series, exog=exog_df, order=(1, 2, 1))
    arimax_fitted = arimax_model.fit(disp=False)
    
    exog_forecast = forecast_exog_arima(exog_df, forecast_years)
    forecast = arimax_fitted.forecast(steps=len(forecast_years), exog=exog_forecast)
    
    return forecast.values
