from sklearn.linear_model import LinearRegression
import numpy as np

def linear_forecast(years, gdp, forecast_years):
    linear_model = LinearRegression().fit(years[1:], gdp[1:])
    preds = linear_model.predict(forecast_years.reshape(-1,1))
    #print(f"Linear equation: GDP = {linear_model.coef_[0]:.1f} * Year + {linear_model.intercept_:.1f}")
    return preds
