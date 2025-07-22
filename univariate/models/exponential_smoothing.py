from statsmodels.tsa.holtwinters import ExponentialSmoothing

def exponential_smoothing(gdp, forecast_years):
    trends = ['add','mul',None]
    best_aic = float('inf')
    best_model = None

    for trend in trends:
        try:
            model = ExponentialSmoothing(gdp, trend=trend, seasonal=None)
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_model = fitted
        except:
            continue

    forecast = best_model.forecast(steps=len(forecast_years))
    return forecast