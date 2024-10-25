import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

st.title("Stock Price Prediction Web App")

# Sidebar
st.sidebar.header('User Input')
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('2024-01-01'))
stock_symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS")

# Load the stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)
data.reset_index(inplace=True)

# Display stock data
st.subheader('Stock Data')
st.write(data.tail())

# Plotting the stock data
st.subheader('Closing Price vs Time Chart')
plt.figure(figsize=(8, 6))
plt.plot(data['Date'], data['Close'], 'g')
plt.xlabel('Date')
plt.ylabel('Closing Price')
st.pyplot(plt)

# Moving averages
st.subheader('Moving Averages')
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], ma_100_days, 'r', label="100-Day Moving Average")
plt.plot(data['Date'], ma_200_days, 'b', label="200-Day Moving Average")
plt.plot(data['Date'], data['Close'], 'g', label="Closing Price")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)

# Prepare the data for prediction
data.dropna(inplace=True)
data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaler = scaler.fit_transform(data_train)

x_train = []
y_train = []

for i in range(100, data_train_scaler.shape[0]):
    x_train.append(data_train_scaler[i-100:i])
    y_train.append(data_train_scaler[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load or build the model
try:
    model = load_model('Stock Price Prediction Model.keras')
    st.success("Model loaded successfully!")
except:
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(units=60, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)
    model.save('Stock Price Prediction Model.keras')
    st.success("Model trained and saved successfully!")

# Preparing the test data for prediction
# Preparing the test data for prediction
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)

# Scaling the test data
data_test_scaler = scaler.transform(data_test)

x_test = []
for i in range(100, data_test_scaler.shape[0]):
    x_test.append(data_test_scaler[i-100:i])

x_test = np.array(x_test)

# Predicting stock prices
y_predicted = model.predict(x_test)

# Rescaling the predicted values back to the original scale
y_predicted_rescaled = scaler.inverse_transform(y_predicted)

# Rescaling the original test data back to the original scale
y_test_rescaled = scaler.inverse_transform(data_test_scaler[100:])

# Create a date index for the test set
test_dates = data['Date'][int(len(data) * 0.80):].reset_index(drop=True)

# Plotting the results: Predicted vs Original Prices
st.subheader('Predicted vs Original Prices')
plt.figure(figsize=(10, 6))
plt.plot(test_dates, y_test_rescaled, 'g', label='Original Price')
plt.plot(test_dates, y_predicted_rescaled, 'r', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)

