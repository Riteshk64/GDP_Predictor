import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Import Model Functions --------------------
from univariate.models.linear_regression import linear_regression
from univariate.models.moving_average import moving_average
from univariate.models.exponential_smoothing import exponential_smoothing
from univariate.models.arima import arima
from multivariate.models.prophet_univariate import prophet_univariate

from multivariate.models.arimax import arimax
from multivariate.models.var import var
from multivariate.models.prophet_multivariate import prophet_multivariate

from prophet.plot import add_changepoints_to_plot
from prophet import Prophet

import warnings
warnings.filterwarnings("ignore")

# -------------------- Page Config --------------------
st.set_page_config(page_title="GDP Forecast Dashboard")

# -------------------- Sidebar Options --------------------
model_type = st.sidebar.selectbox("Model Type", ["Univariate", "Multivariate"])

if model_type == "Univariate":
    model_choice = st.sidebar.selectbox("Model", [
        "Linear", "Moving Average", "Exponential Smoothing", "ARIMA", "Prophet"
    ])
else:
    model_choice = st.sidebar.selectbox("Model", [
        "ARIMAX", "VAR", "Prophet"
    ])

# -------------------- Load Data --------------------
df = pd.read_csv("multivariate/data/processed/gdp_factor_india.csv")
df = df[df['Year'].between(1991, 2025)].reset_index(drop=True)

# For univariate models
df_univariate = df[['Year', 'GDP (current US$)']].copy()
df_univariate = df_univariate.dropna().reset_index(drop=True)
years = np.array(df_univariate["Year"]).reshape(-1, 1)
gdp_univariate = np.array(df_univariate["GDP (current US$)"])
forecast_years_univariate = np.arange(2025, 2031)

# -------------------- Forecast Logic --------------------
forecast = None
fig = None
ci_lower, ci_upper = None, None  # For ARIMA confidence intervals

# ---------- Univariate Models ----------
if model_type == "Univariate":
    if model_choice == "Linear":
        forecast = linear_regression(years, gdp_univariate, forecast_years_univariate)

    elif model_choice == "Moving Average":
        forecast = moving_average(years, gdp_univariate, forecast_years_univariate)

    elif model_choice == "Exponential Smoothing":
        forecast = exponential_smoothing(gdp_univariate, forecast_years_univariate)

    elif model_choice == "ARIMA":
        arima_result = arima(gdp_univariate, forecast_years_univariate)
        if isinstance(arima_result, tuple):
            forecast, arima_ci, _ = arima_result
            ci_lower, ci_upper = arima_ci
        else:
            forecast = arima_result

    elif model_choice == "Prophet":
        forecast = prophet_univariate("multivariate/data/processed/gdp_factor_india.csv", n_years=6)

        # Prepare data for Prophet plotting
        df_prophet = pd.read_csv("multivariate/data/processed/gdp_factor_india.csv")
        df_prophet = df_prophet.rename(columns={"Year": "ds", "GDP (current US$)": "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")

        model = Prophet(
            yearly_seasonality=False,
            daily_seasonality=False,
            weekly_seasonality=False,
            changepoint_prior_scale=0.1,
            n_changepoints=10
        )
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=6, freq="Y")
        forecast_df = model.predict(future)

        fig = model.plot(forecast_df)
        ax = fig.gca()
        add_changepoints_to_plot(ax, model, forecast_df)
        plt.axvline(x=df_prophet["ds"].max(), color="gray", linestyle="--")

# ---------- Multivariate Models ----------
elif model_type == "Multivariate":
    df.set_index('Year', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y')
    df = df.asfreq('YS')

    gdp = df['GDP (current US$)']
    exog = df.drop(columns=['GDP (current US$)'])
    forecast_years = pd.date_range(start='2025', periods=6, freq='YS')

    if model_choice == "ARIMAX":
        arimax_result = arimax(gdp, exog, forecast_years)
        if isinstance(arimax_result, tuple):
            forecast = arimax_result[0]
        else:
            forecast = arimax_result

    elif model_choice == "VAR":
        forecast = var(df, forecast_years)

    elif model_choice == "Prophet":
        forecast = prophet_multivariate("multivariate/data/processed/gdp_factor_india.csv", forecast_periods=6)

        df_mv = pd.read_csv("multivariate/data/processed/gdp_factor_india.csv")
        df_mv['ds'] = pd.to_datetime(df_mv["Year"], format="%Y")
        df_mv['y'] = pd.to_numeric(df_mv['GDP (current US$)'], errors='coerce')
        regressor_cols = [col for col in df_mv.columns if col not in ['Year', 'GDP (current US$)', 'ds', 'y']]
        df_mv[regressor_cols] = df_mv[regressor_cols].apply(pd.to_numeric, errors='coerce')
        df_mv = df_mv.dropna().reset_index(drop=True)

        model = Prophet(
            yearly_seasonality=False,
            changepoint_prior_scale=0.1,
            n_changepoints=10
        )
        for reg in regressor_cols:
            model.add_regressor(reg)

        model.fit(df_mv[['ds', 'y'] + regressor_cols])

        future = model.make_future_dataframe(periods=6, freq="YS")
        for reg in regressor_cols:
            future[reg] = list(df_mv[reg]) + [df_mv[reg].iloc[-1]] * 6
            future[reg] = future[reg][:len(future)]

        forecast_df = model.predict(future)

        fig = model.plot(forecast_df)
        ax = fig.gca()
        add_changepoints_to_plot(ax, model, forecast_df)
        plt.axvline(x=df_mv["ds"].max(), color="gray", linestyle="--")

# -------------------- Plot and Display --------------------
st.header(f"{model_choice} Forecast")

if forecast is not None and fig is None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if model_type == "Univariate":
        ax.plot(df_univariate["Year"], df_univariate["GDP (current US$)"], "o-", label="Historical")
        ax.plot(forecast_years_univariate, forecast, "o--", label="Forecast")

        if model_choice == "ARIMA" and ci_lower is not None and ci_upper is not None:
            ax.fill_between(forecast_years_univariate, ci_lower, ci_upper,
                            color='red', alpha=0.2, label='95% Confidence Interval')

    else:
        ax.plot(df.index.year, gdp, "o-", label="Historical GDP")
        if isinstance(forecast, pd.Series) or isinstance(forecast, pd.DataFrame):
            forecast_values = forecast.values.flatten() if hasattr(forecast, 'values') else forecast
        else:
            forecast_values = forecast
        ax.plot(forecast_years.year, forecast_values, "o--", label=f"{model_choice} Forecast")

    ax.set_xlabel("Year")
    ax.set_ylabel("GDP (in billion dollars)")
    ax.legend()
    ax.grid(True, alpha=0.3)

if fig:
    st.pyplot(fig)

# -------------------- Forecast Table & Export --------------------
if forecast is not None:
    forecast_years_table = forecast_years_univariate if model_type == "Univariate" else forecast_years.year

    forecast_display = pd.DataFrame({
        "Year": forecast_years_table,
        "Forecasted GDP": np.round(forecast, 2)
    })

    if model_choice == "ARIMA" and ci_lower is not None and ci_upper is not None:
        forecast_display["Lower Bound (95% CI)"] = np.round(ci_lower, 2)
        forecast_display["Upper Bound (95% CI)"] = np.round(ci_upper, 2)

    st.dataframe(forecast_display)

    csv = forecast_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"gdp_forecast_{model_choice.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv",
        mime="text/csv"
    )
