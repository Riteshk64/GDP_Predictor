import pandas as pd
from prophet import Prophet

def prophet_univariate(csv_path, n_years=6):
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Drop unwanted columns if they exist
        drop_cols = ['Per_Capita_in_USD', 'Percentage_Growth']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # Rename columns for Prophet: 
        # Prophet expects 'ds' for date/time and 'y' for the target variable
        # Assuming 'Year' column contains the year and 'GDP (current US$)' is target
        df = df.rename(columns={
            'Year': 'ds',
            'GDP (current US$)': 'y'
        })

        # Convert ds column to datetime (year only)
        df['ds'] = pd.to_datetime(df['ds'], format='%Y')

        df = df.sort_values('ds').reset_index(drop=True)

        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality=False,
            daily_seasonality=False,
            weekly_seasonality=False,
            changepoint_prior_scale=0.5,
            n_changepoints=10
        )
        model.fit(df)

        # Create future dataframe for forecasting
        future = model.make_future_dataframe(periods=n_years, freq='Y')
        forecast = model.predict(future)

        # Return only forecasted yhat values for the future periods
        forecast_df = forecast[['ds', 'yhat']].tail(n_years).reset_index(drop=True)
        return forecast_df['yhat'].values

    except Exception as e:
        print(f"Prophet univariate forecast failed: {e}")
        return None
