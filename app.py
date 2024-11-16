import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st
from datetime import date, timedelta, datetime


end = date.today() - timedelta(days = 1)
start = date.today() - timedelta(days = 1) - timedelta(weeks = 522)

st.set_page_config(page_title="Simple LSTM Stock Prediction")
st.title('Stock Trend Prediction')

stock_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(stock_input, start, end)

#Data Description
st.subheader('Data From %s - %s'%(str(start.year),str(end.year)))
st.write(df.describe())

#Data Visualizations
st.subheader('Closing Price v/s Time Chart')
fig = plt.figure(figsize= (12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price v/s Time Chart with 100 Day MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize= (12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader('Closing Price v/s Time Chart with 100 & 200 Day MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize= (12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#Load the Model
model = load_model('keras_stock_model.keras')

previous_100_days = data_training.tail(100)
final_testing = pd.concat([previous_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_testing)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scale_factor = 1/scaler.scale_[0]

y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize= (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Predict for the next 10 days
st.subheader('Predicted Closing Prices for Next 10 Days')

last_100_days = final_testing[-100:].values
last_100_scaled = scaler.transform(last_100_days)

future_predictions = []
current_input = last_100_scaled[-100:].reshape(1, -1, 1)

for _ in range(10):
    next_pred = model.predict(current_input)[0][0]
    future_predictions.append(next_pred)

    # Update the current input by appending the prediction and removing the first value
    next_input = np.append(current_input[0][1:], [[next_pred]], axis=0)
    current_input = next_input.reshape(1, -1, 1)

# Rescale predictions back to original scale
future_predictions = np.array(future_predictions) * scale_factor

# Display future predictions
future_dates = [end + timedelta(days=i+1) for i in range(10)]
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

st.write(future_df)

# Plot future predictions
fig3 = plt.figure(figsize=(12, 6))
plt.plot(future_dates, future_predictions, marker='o', linestyle='-', color='purple', label='Future Predictions')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.legend()
st.pyplot(fig3)