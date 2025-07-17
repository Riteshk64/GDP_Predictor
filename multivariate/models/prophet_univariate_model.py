import pandas as pd
from prophet import Prophet

def prophet_univariate_forecast(csv_path, n_years=9):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.drop(['Per_Capita_in_USD', 'Percentage_Growth'], axis=1)
    df = df.rename(columns={'Year': 'ds', 'GDP_In_Billion_USD': 'y'})
    df['ds'] = pd.to_datetime(df['ds'], format='%Y')
    df = df.sort_values('ds').reset_index(drop=True)

    model = Prophet(
        yearly_seasonality=False,
        daily_seasonality=False,
        weekly_seasonality=False,
        changepoint_prior_scale=0.1,
        n_changepoints=10
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=n_years, freq='Y')
    forecast = model.predict(future)

    forecast_df = forecast[['ds', 'yhat']].tail(n_years).reset_index(drop=True)
    return forecast_df['yhat'].values
