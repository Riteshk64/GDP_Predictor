import numpy as np
from sklearn.linear_model import LinearRegression

def linear_regression(years, gdp, forecast_years):
    best_score = -float('inf')
    best_model = None
    best_degree = 1

    for degree in [1, 2, 3]:
        try:
            X = np.column_stack([years**i for i in range(1, degree+1)])
            model = LinearRegression().fit(X, gdp)
            score = model.score(X, gdp)
            if score > best_score:
                best_score = score
                best_model = model
                best_degree = degree
        except:
            continue

    X_forecast = np.column_stack([forecast_years**i for i in range(1, best_degree+1)])
    forecast = best_model.predict(X_forecast)
    return forecast