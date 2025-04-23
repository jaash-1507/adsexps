#pip install numpy pandas matplotlib scipy statsmodels

import numpy as np 
import pandas as pd 
from scipy import stats 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose 

# Load Air Passengers dataset
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
                   parse_dates=['Month'], index_col='Month')
data.rename(columns={'Passengers': 'values'}, inplace=True)

# Handle missing values (if any)
data['values'].interpolate(inplace=True)

# Prepare time and value series
time = list(range(len(data.index)))
values = data['values']

# Performing decomposition
decomposition = seasonal_decompose(values, model='additive', period=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Display original data
print("Original Data:")
print(data.head())

# Plotting the decomposition
fig, ax = plt.subplots(4, figsize=(10, 8))

ax[0].plot(time, values, label='Original Data')
ax[0].set_title("Original Data")

ax[1].plot(time, trend, label='Trend', color='orange')
ax[1].set_title("Trend")

ax[2].plot(time, seasonal, label='Seasonality', color='green')
ax[2].set_title("Seasonality")

ax[3].plot(time, residual, label='Residual', color='red')
ax[3].set_title("Residual")

for axis in ax:
    axis.legend()
    axis.grid(True)

plt.tight_layout()
plt.show()
