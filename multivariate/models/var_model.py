import pandas as pd
from statsmodels.tsa.api import VAR

def var_forecast(df, forecast_years, maxlags=1):
    var_model = VAR(df)
    var_fitted = var_model.fit(maxlags=maxlags, trend='ct')
    
    forecast = var_fitted.forecast(df.values, steps=len(forecast_years))
    forecast_df = pd.DataFrame(forecast, columns=df.columns, index=forecast_years)
    
    return forecast_df['GDP (current US$)'].values
