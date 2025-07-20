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

# for univariate models
df_univariate = df[['Year', 'GDP (current US$)']].copy()
df_univariate = df_univariate.dropna().reset_index(drop=True)
years = np.array(df_univariate["Year"]).reshape(-1, 1)
gdp = np.array(df_univariate["GDP (current US$)"])
forecast_years = np.arange(2025, 2031)

# -------------------- Forecast Logic --------------------
forecast = None
fig = None

# ---------- Univariate Models ----------
if model_type == "Univariate":
    if model_choice == "Linear":
        forecast = linear_regression(years, gdp, forecast_years)

    elif model_choice == "Moving Average":
        forecast = moving_average(gdp, forecast_years)

    elif model_choice == "Exponential Smoothing":
        forecast = exponential_smoothing(gdp, forecast_years)

    elif model_choice == "ARIMA":
        forecast = arima(gdp, forecast_years)

    elif model_choice == "Prophet":
        forecast = prophet_univariate("multivariate/data/univariate_gdp.csv", n_years=6)

        # Optional Prophet plot
        df_prophet = df.rename(columns={"Year": "ds", "GDP_In_Billion_USD": "y"})
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
        plt.title("Univariate Prophet GDP Forecast")

# ---------- Multivariate Models ----------
elif model_type == "Multivariate":
    if model_choice == "ARIMAX":
        forecast = arimax(gdp, exog, forecast_years)

    elif model_choice == "VAR":
        forecast = var(df, forecast_years)

    elif model_choice == "Prophet (Multivariate)":
        forecast = prophet_multivariate("multivariate/data/processed/gdp_factor_india.csv", forecast_periods=6)

        # Optional Prophet plot
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
        future = model.make_future_dataframe(periods=6, freq="Y")
        for reg in regressor_cols:
            future[reg] = pd.concat([df_mv[reg], pd.Series([df_mv[reg].iloc[-1]] * 6)], ignore_index=True)

        forecast_df = model.predict(future)

        fig = model.plot(forecast_df)
        ax = fig.gca()
        add_changepoints_to_plot(ax, model, forecast_df)
        plt.axvline(x=df_mv["ds"].max(), color="gray", linestyle="--")
        plt.title("Multivariate Prophet GDP Forecast")

# -------------------- Plot and Display --------------------
st.header(f"{model_choice} Forecast")

if forecast is not None and fig is None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if model_type == "Univariate":
        ax.plot(df["Year"], df["GDP_In_Billion_USD"], "o-", label="Historical")
        ax.plot(forecast_years, forecast, "o--", label="Forecast")
    else:
        ax.plot(df.index, gdp, "o-", label="Historical GDP")
        ax.plot(forecast_years, forecast, "o--", label=f"{model_choice} Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP (in billion dollars)")
    ax.legend()
    ax.grid(True, alpha=0.3)

if fig:
    st.pyplot(fig)

# -------------------- Forecast Table & Export --------------------
if forecast is not None:
    forecast_display = pd.DataFrame({
        "Year": forecast_years.year if model_type == "Multivariate" else forecast_years,
        "Forecasted GDP": np.round(forecast, 2)
    })
    st.dataframe(forecast_display)

    csv = forecast_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"gdp_forecast_{model_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )
