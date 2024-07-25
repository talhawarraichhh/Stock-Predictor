from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

def predict_stock(ticker_symbol):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2018-01-01'

    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    data = stock_data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data)-time_step-1):
            a = data[i:(i+time_step), 0]
            X.append(a)
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[0:train_size]
    test_data = scaled_data[train_size:len(scaled_data)]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=64, epochs=10)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    combined_data = np.concatenate((train_data, test_data), axis=0)
    last_sequence = combined_data[-time_step:]
    future_predictions = []

    for _ in range(30):
        next_pred = model.predict(last_sequence.reshape(1, time_step, 1))
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    last_known_price = scaler.inverse_transform(scaled_data[-1].reshape(1, -1))[0][0]
    percentage_change = ((future_predictions[-1][0] - last_known_price) / last_known_price) * 100
    percentage_change = round(percentage_change, 2)

    last_date = pd.to_datetime(stock_data.index[-1])
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

    dates = pd.to_datetime(stock_data.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=scaler.inverse_transform(scaled_data).flatten(), mode='lines', name='Actual Stock Price', line=dict(color='magenta')))
    fig.add_trace(go.Scatter(x=dates[time_step:len(train_predict) + time_step], y=train_predict.flatten(), mode='lines', name='Train Predict', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=dates[len(train_predict) + (time_step * 2) + 1:len(train_predict) + (time_step * 2) + 1 + len(test_predict)], y=test_predict.flatten(), mode='lines', name='Test Predict', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Future Predictions', line=dict(color='cyan', dash='dash')))

    fig.update_layout(
        title='Predicted Stock Prices',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        hovermode='x unified',
        paper_bgcolor='#252525',
        plot_bgcolor='#252525',
        font=dict(color='white')
    )

    graph_html = pio.to_html(fig, full_html=False)

    return graph_html, future_predictions, percentage_change

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    graph_html, future_predictions, percentage_change = predict_stock(ticker)
    return render_template('result.html', ticker=ticker, graph_html=graph_html, predictions=future_predictions, percentage_change=percentage_change)

if __name__ == '__main__':
    app.run(debug=True)
