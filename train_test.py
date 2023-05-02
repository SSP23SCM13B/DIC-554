import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Read the CSV file
df = pd.read_csv("updated_bitstamp.csv", index_col=0, parse_dates=True)

# Create a copy of the DataFrame
bitstamp2 = df.copy()

# Concatenate the two DataFrames
bitstamps = pd.concat([bitstamp2, df])

# Resample the data to daily frequency
bitstamp_daily = bitstamps.resample("24H").mean()

# Get the Weighted_Price column and convert it to a NumPy array
price_series = bitstamp_daily.reset_index().Weighted_Price.values

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
price_series_scaled = scaler.fit_transform(price_series.reshape(-1,1))

# Split the data into training and test sets
train_data, test_data = price_series_scaled[0:2923], price_series_scaled[2923:]

# Define a function to create windowed dataset
def windowed_dataset(series, time_step):
    dataX, dataY = [], []
    for i in range(len(series)- time_step-1):
        a = series[i : (i+time_step), 0]
        dataX.append(a)
        dataY.append(series[i+ time_step, 0])
    return np.array(dataX), np.array(dataY)

# Create the training and test datasets
X_train, y_train = windowed_dataset(train_data, time_step=100)
X_test, y_test = windowed_dataset(test_data, time_step=100)

# Reshape the datasets to fit the input shape of the LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create the LSTM model
regressor = Sequential()

# Add the first LSTM layer with dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Add the second LSTM layer with dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Add the third LSTM layer with dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Add the fourth LSTM layer with dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Add the output layer
regressor.add(Dense(units=1))

# Compile the model
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model to the training data
history = regressor.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, verbose=1, shuffle=False)

# Make predictions on the training and test datasets
train_predict = regressor.predict(X_train)
test_predict = regressor.predict(X_test)

# Inverse transform the predictions and actual values to the original scale
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
train_predict_inv = scaler.inverse_transform(train_predict)
test_predict_inv = scaler.inverse_transform(test_predict)