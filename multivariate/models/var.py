import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

def var(df, forecast_years, maxlags=2):
    try:
        if df.isnull().values.any():
            df = df.dropna()
        
        max_possible_lags = max(1, int(len(df) / (len(df.columns) + 1)))
        maxlags = min(maxlags, max_possible_lags)

        def is_stationary(series):
            result = adfuller(series)
            return result[1] < 0.05

        for col in df.columns:
            if not is_stationary(df[col]):
                print(f"[INFO] Column '{col}' not stationary. Consider differencing.")

        var_model = VAR(df)
        selected_order = var_model.select_order(maxlags=maxlags)
        aic = selected_order.aic
        best_lag = aic.idxmin() if hasattr(aic, 'idxmin') else aic
        print(f"[INFO] Best lag order (AIC): {best_lag}")

        var_fitted = var_model.fit(best_lag, trend='ct')

        forecast = var_fitted.forecast(df.values, steps=len(forecast_years))
        forecast_df = pd.DataFrame(forecast, columns=df.columns, index=forecast_years)

        return forecast_df['GDP (current US$)'].values

    except Exception as e:
        print(f"[ERROR] VAR model fitting failed: {e}")
        return np.repeat(df['GDP (current US$)'].iloc[-1], len(forecast_years))
