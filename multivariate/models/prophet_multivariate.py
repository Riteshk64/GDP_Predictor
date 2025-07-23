import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Optional: Silence ARIMA warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)


def forecast_regressors(exog_df, forecast_periods):
    future_values = {}
    for col in exog_df.columns:
        try:
            series = exog_df[col].dropna().reset_index(drop=True)

            if len(series) < 5:
                raise ValueError("Too few observations to model")

            model = ARIMA(series, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
            forecast = fitted.forecast(steps=forecast_periods)
            future_values[col] = forecast.values

        except Exception as e:
            print(f"[WARN] Forecast failed for '{col}': {e}")
            future_values[col] = [series.iloc[-1]] * forecast_periods

    return pd.DataFrame(future_values)


def prophet_multivariate(df_path, forecast_periods=6):
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
        changepoint_prior_scale=0.5,
        n_changepoints=10
    )

    for reg in regressor_cols:
        model.add_regressor(reg)

    fit_df = df[['ds', 'y'] + regressor_cols].dropna()
    model.fit(fit_df)

    # Create full-length future DataFrame
    future = model.make_future_dataframe(periods=forecast_periods, freq='Y')

    # Forecast regressors
    historical_regressors = fit_df[regressor_cols]
    exog_forecast = forecast_regressors(historical_regressors, forecast_periods)

    # Concatenate historical + forecast regressors
    full_regressor_df = pd.concat([historical_regressors, exog_forecast], ignore_index=True)
    full_regressor_df.index = future.index

    # Add to future dataframe
    for reg in regressor_cols:
        future[reg] = full_regressor_df[reg].values

    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(forecast_periods)['yhat'].values
