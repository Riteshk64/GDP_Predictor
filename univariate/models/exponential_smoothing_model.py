from statsmodels.tsa.holtwinters import ExponentialSmoothing

def exponential_smoothing_forecast(gdp, forecast_years):
    es_model = ExponentialSmoothing(gdp, trend='add', seasonal=None)
    es_fitted = es_model.fit()
    return es_fitted.forecast(steps=len(forecast_years))
