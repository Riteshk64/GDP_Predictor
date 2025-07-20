import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

def forecast_regressors(exog_df, forecast_periods):
    future_values = {}
    for col in exog_df.columns:
        series = exog_df[col]
        model = ARIMA(series, order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_periods)
        future_values[col] = forecast.values
    return pd.DataFrame(future_values)

def prophet_multivariate(df_path, forecast_periods=9):
    df = pd.read_csv(df_path)
    df['ds'] = pd.to_datetime(df['Year'], format='%Y')
    df['y'] = pd.to_numeric(df['GDP (current US$)'], errors='coerce')

    regressor_cols = [
        'Current account balance (% of GDP)',
        'Exports of goods and services (% of GDP)',
        'Final consumption expenditure (% of GDP)',
        'Imports of goods and services (% of GDP)',
        'Industry (including construction), value added (% of GDP)',
        'Inflation, consumer prices (annual %)',
        'Gross capital formation (% of GDP)',
    ]

    model = Prophet(
        yearly_seasonality=False,
        daily_seasonality=False,
        weekly_seasonality=False,
        changepoint_prior_scale=0.1,
        n_changepoints=10
    )

    for reg in regressor_cols:
        model.add_regressor(reg)

    fit_df = df[['ds', 'y'] + regressor_cols]
    model.fit(fit_df)

    # Make future dataframe for target GDP
    future = model.make_future_dataframe(periods=forecast_periods, freq='Y')

    # Forecast future regressors using ARIMA for each column
    historical_regressors = df[regressor_cols]
    exog_forecast = forecast_regressors(historical_regressors, forecast_periods)

    # Concatenate historical and forecasted regressors
    for reg in regressor_cols:
        future[reg] = pd.concat(
            [historical_regressors[reg], exog_forecast[reg]],
            ignore_index=True
        )

    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(forecast_periods)['yhat'].values