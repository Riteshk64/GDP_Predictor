import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def forecast_exog_arima(exog_df, forecast_years, order=(1, 0, 0)):
    exog_forecast = pd.DataFrame(index=forecast_years, columns=exog_df.columns)

    for col in exog_df.columns:
        try:
            series = exog_df[col].dropna()
            if len(series) < 5:
                raise ValueError("Insufficient data for ARIMA")

            model = ARIMA(series, order, enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(forecast_years))
            exog_forecast[col] = pd.Series(forecast.values, index=forecast_years)
        except Exception as e:
            print(f"[WARN] Forecast failed for '{col}': {e}")
            exog_forecast[col] = [series.iloc[-1]] * len(forecast_years)

    return exog_forecast
