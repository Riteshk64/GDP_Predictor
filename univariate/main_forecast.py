import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

from models.arima import arima
from models.exponential_smoothing import exponential_smoothing
from models.linear_regression import linear_regression
from models.moving_average import moving_average

warnings.filterwarnings('ignore')

# Load the comprehensive dataset
def load_and_prepare_data():
    """Load and prepare GDP data from the factors dataset"""
    df = pd.read_csv('multivariate/data/processed/gdp_factor_india.csv')
    df = df.sort_values('Year').reset_index(drop=True)

    gdp_billions = df['GDP (current US$)']
    years = df['Year'].values
    
    return years, gdp_billions.values

# Model evaluation metrics
def calculate_metrics(actual, predicted):
    """Calculate comprehensive evaluation metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Main forecasting function
def main_forecast():
    """Main function to run all improved models"""
    
    # Load data
    years, gdp = load_and_prepare_data()
    forecast_years = np.array([2025, 2026, 2027, 2028, 2029, 2030])
    
    # Run all models
    all_forecasts = {}
    confidence_intervals = {}
    
    # ARIMA Model
    arima_forecast, arima_ci, arima_order = arima(gdp, forecast_years)
    all_forecasts['ARIMA'] = arima_forecast
    confidence_intervals['ARIMA'] = arima_ci
    
    # Exponential Smoothing
    all_forecasts['Exponential Smoothing'] = exponential_smoothing(gdp, forecast_years)
    
    # Linear Regression
    all_forecasts['Linear Regression'] = linear_regression(years, gdp, forecast_years)
    
    # Moving Average
    all_forecasts['Moving Average'] = moving_average(years, gdp, forecast_years)
    
    comparison_df = pd.DataFrame({
        'Year': forecast_years,
        'ARIMA': [f'${x:.0f}B' for x in all_forecasts['ARIMA']],
        'Exp Smoothing': [f'${x:.0f}B' for x in all_forecasts['Exponential Smoothing']],
        'Linear Reg': [f'${x:.0f}B' for x in all_forecasts['Linear Regression']],
        'Moving Avg': [f'${x:.0f}B' for x in all_forecasts['Moving Average']]
    })
    
    ensemble_forecast = np.mean([all_forecasts[model] for model in all_forecasts.keys()], axis=0)
    all_forecasts['Ensemble'] = ensemble_forecast

    # Visualization
    plt.figure(figsize=(16, 12))
    
    # Main comparison plot
    plt.subplot(2, 2, 1)
    plt.plot(years, gdp, 'o-', label='Historical GDP', linewidth=3, markersize=4, color='black')
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    models = list(all_forecasts.keys())
    
    for model, color in zip(models, colors):
        plt.plot(forecast_years, all_forecasts[model], 'o-', 
                label=model, color=color, alpha=0.8, linewidth=2)
    
    if 'ARIMA' in confidence_intervals:
        ci_lower, ci_upper = confidence_intervals['ARIMA']
        plt.fill_between(forecast_years, ci_lower, ci_upper, 
                        alpha=0.2, color='red', label='ARIMA 95% CI')
    
    plt.title('India GDP Forecast: All Models Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('GDP (Billion USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Forecast comparison only
    plt.subplot(2, 2, 2)
    for model, color in zip(models, colors):
        plt.plot(forecast_years, all_forecasts[model], 'o-', 
                label=model, color=color, linewidth=2, markersize=6)
    
    plt.title('Forecast Comparison (2025-2030)', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('GDP (Billion USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2030 predictions bar chart
    plt.subplot(2, 2, 3)
    forecast_2030 = [all_forecasts[model][-1] for model in models]
    bars = plt.bar(models, forecast_2030, color=colors, alpha=0.7)
    plt.title('2030 GDP Predictions by Model', fontsize=14, fontweight='bold')
    plt.ylabel('GDP (Billion USD)')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars, forecast_2030):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'${value:.0f}B', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return all_forecasts, confidence_intervals

if __name__ == "__main__":
    forecasts, confidence_intervals = main_forecast()