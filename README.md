# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Path Finder Logic
file_path = None
for path in ['/content/aapl_master_enriched.csv', 'aapl_master_enriched.csv', '/aapl_master_enriched.csv']:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    print("Error: 'aapl_master_enriched.csv' not found. Please upload the file.")
else:
    # 2. Load and Explore Dataset
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    # Using last 500 days for clear time series analysis
    data = df['close'].tail(500)
    
    print("--- OUTPUT: DATA EXPLORATION ---")
    print(data.head())
    
    # 3. Check for Stationarity (ADF Test)
    print("\n--- STATIONARITY CHECK (ADF TEST) ---")
    result = adfuller(data)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    
    # 4. Transform to Stationary (Differencing)
    # If p-value > 0.05, data is non-stationary. We apply 1st order differencing.
    data_diff = data.diff().dropna()

    # 5. Plotting ACF and PACF to determine p and q
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(data_diff, ax=axes[0], lags=30, title='ACF (Apple Stock Differenced)')
    plot_pacf(data_diff, ax=axes[1], lags=30, title='PACF (Apple Stock Differenced)')
    plt.show()

    # 6. Fit ARIMA Model
    # Split: Train on 90%, Test on last 10%
    train_size = int(len(data) * 0.9)
    train, test = data[0:train_size], data[train_size:len(data)]
    
    # Fitting ARIMA(5, 1, 0) - common baseline for stock data
    model = ARIMA(train, order=(5, 1, 0)).fit()
    print("\n--- MODEL SUMMARY ---")
    print(model.summary())

    # 7. Make Predictions
    forecast = model.forecast(steps=len(test))
    
    # 8. Evaluate Model Predictions
    mse = mean_squared_error(test, forecast)
    rmse = np.sqrt(mse)
    
    print("\n--- OUTPUT: FINAL EVALUATION ---")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")

    # 9. Plotting Final Results
    plt.figure(figsize=(12, 6))
    plt.plot(train.tail(100), label='Training Data (Recent)')
    plt.plot(test, label='Actual Apple Price', color='black', linewidth=1.5)
    plt.plot(test.index, forecast, label='ARIMA Forecast', color='red', linestyle='--', linewidth=2)
    plt.title('Apple Stock Price Prediction: ARIMA Model Evaluation')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nRESULT: ARIMA model implemented and evaluated successfully.")
    
```

### OUTPUT:

<img width="776" height="414" alt="image" src="https://github.com/user-attachments/assets/a00c201a-9a5c-4603-93f9-52df51dd7d2a" />
<img width="414" height="308" alt="image" src="https://github.com/user-attachments/assets/f1a6917b-8e01-4957-9d9f-96f468ffc09a" />
<img width="653" height="404" alt="image" src="https://github.com/user-attachments/assets/2ae2bbfe-0471-4866-a052-b4cd18d4073b" />

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
