# Ex.No: 6               HOLT WINTERS METHOD

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
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_excel("/content/Large_Yearly_Sales_Data_1900_2024.xlsx")

data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

data.head()

data_yearly = data['Sales_Rate']

data_yearly.plot()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_yearly.values.reshape(-1, 1)).flatten(),
    index=data_yearly.index
)

scaled_data.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_yearly, model="additive")
decomposition.plot()
plt.show()

scaled_data = scaled_data + 1
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal=None).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual evaluation')

print(np.sqrt(mean_squared_error(test_data, test_predictions_add)))

final_model = ExponentialSmoothing(data_yearly, trend='add', seasonal=None).fit()

final_predictions = final_model.forecast(steps=10)

ax = data_yearly.plot()
final_predictions.plot(ax=ax)

ax.legend(["data_yearly", "final_predictions"])
ax.set_xlabel('Year')
ax.set_ylabel('Sales_Rate')
ax.set_title('Prediction')

plt.show()
```

### OUTPUT:
<img width="466" height="358" alt="image" src="https://github.com/user-attachments/assets/b709651d-27dc-469d-b40e-dc5044a14eac" />


TEST_PREDICTION

<img width="522" height="401" alt="image" src="https://github.com/user-attachments/assets/5a11bd84-e575-4a76-99f6-3539921ff2b6" />


FINAL_PREDICTION

<img width="479" height="372" alt="image" src="https://github.com/user-attachments/assets/0b2fdcf9-8f57-4c05-8800-714623413bc6" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
