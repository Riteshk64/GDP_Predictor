import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from .exog_forecast_model import forecast_exog_arima

def determine_d(series, max_diff=3):
    for d in range(max_diff + 1):
        test_series = np.diff(series, n=d) if d > 0 else series
        if len(test_series) > 10:
            adf_result = adfuller(test_series, maxlag=min(12, len(test_series)//4))
            if adf_result[1] < 0.05:
                return d
    return max_diff

def arimax(gdp_series, exog_df, forecast_years):
    d = determine_d(gdp_series)

    best_aic = float('inf')
    best_order = None

    # Search over small grid of p and q
    for p in range(0, 3):
        for q in range(0, 3):
            try:
                model = SARIMAX(gdp_series, exog=exog_df, order=(p,d,q))
                fitted = model.fit(disp=False)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p,d,q)
            except:
                continue

    # Forecast exogenous variables into future
    exog_forecast = forecast_exog_arima(exog_df, forecast_years, best_order)

    # Fit final model on full data
    final_model = SARIMAX(gdp_series, exog=exog_df, order=best_order)
    final_fitted = final_model.fit(disp=False)

    forecast = final_fitted.forecast(steps=len(forecast_years), exog=exog_forecast)
    return forecast.values