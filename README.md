# Ex.No: 6               HOLT WINTERS METHOD
### Date: 10-03-2026

### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions

### PROGRAM:

```
Name : Prakash C
Reg.No : 212223240122
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

data = pd.read_csv('NSE-TATAGLOBAL11.csv', parse_dates=['Date'], index_col='Date')
print(data.head())
print(data.info())

data = data[['Close']]

data_monthly = data.resample('MS').mean()
print(data_monthly.head())

data_monthly.plot(figsize=(10,5), title="Monthly Close Price")
plt.show()

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(data_monthly.values)
scaled_data = pd.Series(
    scaled_values.flatten(),
    index=data_monthly.index
)

scaled_data.plot(title="Scaled Data")
plt.show()

decomposition = seasonal_decompose(data_monthly, model='additive')

decomposition.plot()
plt.show()

scaled_data = scaled_data + 1   # for multiplicative seasonality

train_size = int(len(scaled_data) * 0.8)

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot(figsize=(10,5))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend(["Train Data","Predictions","Test Data"])
ax.set_title("Visual Evaluation")

plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("RMSE:", rmse)

final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

future_steps = int(len(data_monthly)/4)
final_predictions = final_model.forecast(steps=future_steps)

ax = scaled_data.plot(figsize=(10,5))
final_predictions.plot(ax=ax)
ax.legend(["Historical Data","Future Predictions"])
ax.set_title("Stock Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Scaled Close Price")


```

### OUTPUT:


#### TEST_PREDICTION

<img width="724" height="646" alt="image" src="https://github.com/user-attachments/assets/aaa4fd48-39c1-4e48-85f9-ce16420fe404" />

<img width="234" height="163" alt="image" src="https://github.com/user-attachments/assets/0e0c31b4-8715-4982-a39d-b0bd0b392f2e" />

<img width="1109" height="591" alt="image" src="https://github.com/user-attachments/assets/d0325ac5-e36e-4a3f-bcf7-5e2cd25773d4" />

<img width="866" height="602" alt="image" src="https://github.com/user-attachments/assets/72afd517-3d9d-4597-b7be-25e2a109f274" />

<img width="1148" height="626" alt="image" src="https://github.com/user-attachments/assets/726aa023-1f8c-4758-8bb1-9e228653251e" />


#### FINAL_PREDICTION

<img width="1113" height="623" alt="image" src="https://github.com/user-attachments/assets/12d0ea89-a7ef-412c-84cc-80864de2125e" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
