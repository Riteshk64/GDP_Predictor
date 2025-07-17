import pandas as pd
from prophet import Prophet

def prophet_multivariate_forecast(df_path, forecast_periods=9):
    df = pd.read_csv(df_path, quotechar='"')
    df = df[(df['Year'] >= 1991) & (df['Year'] <= 2022)].reset_index(drop=True)

    df['ds'] = pd.to_datetime(df['Year'], format='%Y')
    df['y'] = pd.to_numeric(df['GDP (current US$)'], errors='coerce')

    regressor_cols = [
        'Current account balance (% of GDP)',
        'Official exchange rate (LCU per US$, period average)',
        'Exports of goods and services (% of GDP)',
        'General government final consumption expenditure (% of GDP)',
        'Final consumption expenditure (% of GDP)',
        'Imports of goods and services (% of GDP)',
        'Industry (including construction), value added (% of GDP)',
        'Inflation, consumer prices (annual %)',
        'Lending interest rate (%)',
        'Gross capital formation (% of GDP)',
        'Broad money (% of GDP)',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)'
    ]

    df[regressor_cols] = df[regressor_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['y'] + regressor_cols).reset_index(drop=True)

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

    future = model.make_future_dataframe(periods=forecast_periods, freq='Y')
    for reg in regressor_cols:
        last_val = df[reg].iloc[-1]
        future[reg] = pd.concat([df[reg], pd.Series([last_val]*forecast_periods)], ignore_index=True)

    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(forecast_periods)['yhat'].values
