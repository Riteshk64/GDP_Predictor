import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def forecast_exog_arima(exog_df, forecast_years, order=(1, 1, 1)):
    exog_forecast = pd.DataFrame(index=forecast_years, columns=exog_df.columns)

    for col in exog_df.columns:
        try:
            series = exog_df[col]
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(forecast_years))
            exog_forecast[col] = forecast.values
        except Exception as e:
            print(f"Forecast failed for column '{col}': {e}")
            exog_forecast[col] = [series.iloc[-1]] * len(forecast_years)

    return exog_forecast
