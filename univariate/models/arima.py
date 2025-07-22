import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error

def arima(gdp, forecast_years, validation_split=0.8):
    split_idx = int(len(gdp) * validation_split)
    train_gdp = gdp[:split_idx]
    test_gdp = gdp[split_idx:]

    def determine_d(series, max_diff=2):
        for d in range(max_diff + 1):
            test_series = np.diff(series, n=d) if d > 0 else series
            if len(test_series) > 10:
                adf_test = adfuller(test_series, maxlag=min(12, len(test_series)//4))
                if adf_test[1] < 0.05:
                    return d
        return max_diff

    d = determine_d(train_gdp)

    best_aic = float('inf')
    best_order = None
    best_model = None

    for p in range(0, 4):
        for q in range(0, 4):
            try:
                model = ARIMA(train_gdp, order=(p, d, q))
                fitted = model.fit()

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_model = fitted
            except:
                continue

    final_model = ARIMA(gdp, order=best_order).fit()
    forecast = final_model.forecast(steps=len(forecast_years))

    forecast_result = final_model.get_forecast(steps=len(forecast_years))
    forecast_ci = forecast_result.conf_int()
    if hasattr(forecast_ci, 'iloc'):
        ci_lower = forecast_ci.iloc[:, 0].values
        ci_upper = forecast_ci.iloc[:, 1].values
    else:
        ci_lower = forecast_ci[:, 0]
        ci_upper = forecast_ci[:, 1]

    return forecast, (ci_lower, ci_upper), best_order