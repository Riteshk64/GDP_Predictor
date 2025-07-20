import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

def var(df, forecast_years, maxlags=4):
    try:
        var_model = VAR(df)
        # Select best lag order (up to maxlags)
        selected_order = var_model.select_order(maxlags=maxlags)
        best_lag = selected_order.aic.idxmin()
        
        var_fitted = var_model.fit(best_lag, trend='ct')
        
        forecast = var_fitted.forecast(df.values, steps=len(forecast_years))
        forecast_df = pd.DataFrame(forecast, columns=df.columns, index=forecast_years)
        
        return forecast_df['GDP (current US$)'].values
    except Exception as e:
        print(f"VAR model fitting failed: {e}")
        return np.repeat(df['GDP (current US$)'].iloc[-1], len(forecast_years))