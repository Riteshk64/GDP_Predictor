from statsmodels.tsa.arima.model import ARIMA

def arima(gdp, forecast_years):
    d = 3
    best_aic = float('inf')
    best_order = None

    for p in range(0, 4):
        for q in range(0, 4):
            try:
                model = ARIMA(gdp, order=(p, d, q))
                fitted = model.fit()

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
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