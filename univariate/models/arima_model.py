import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def arima_forecast(gdp, forecast_years):
    # ADF test to determine differencing order
    def determine_d(series, max_diff=2):
        for d in range(max_diff + 1):
            test_series = np.diff(series, n=d) if d > 0 else series
            adf_test = adfuller(test_series)
            if adf_test[1] < 0.05:
                return d
        return max_diff

    d = determine_d(gdp)
    print(f"\nSelected differencing order (d) based on ADF test: {d}")

    best_aic = float('inf')
    best_order = None
    best_model = None

    for p in [1, 2]:
        for q in [1, 2]:
            try:
                model = ARIMA(gdp, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_model = fitted
            except:
                continue

    print(f"Best ARIMA parameters: {best_order}")
    print(f"Best AIC: {best_aic:.2f}")

    forecast = best_model.forecast(steps=len(forecast_years))
    return forecast
