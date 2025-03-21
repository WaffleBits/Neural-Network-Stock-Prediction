import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class StockPredictor:
    def __init__(self, symbol, start_date='2010-01-01'):
        self.symbol = symbol
        self.start_date = start_date
        self.model = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self):
        data = yf.download(self.symbol, start=self.start_date)
        return data
    
    def prepare_data(self, data, sequence_length=60):
        # Prepare the data for LSTM
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i])
        
        X, y = np.array(X), np.array(y)
        
        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, sequence_length=60):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history
    
    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def plot_predictions(self, actual_values, predicted_values):
        plt.figure(figsize=(16,8))
        plt.plot(actual_values, label='Actual')
        plt.plot(predicted_values, label='Predicted')
        plt.title(f'{self.symbol} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

def main():
    # Example usage
    predictor = StockPredictor('AAPL')
    data = predictor.fetch_data()
    
    X_train, X_test, y_train, y_test = predictor.prepare_data(data)
    predictor.build_model()
    history = predictor.train(X_train, y_train)
    
    predictions = predictor.predict(X_test)
    actual_values = predictor.scaler.inverse_transform(y_test)
    
    predictor.plot_predictions(actual_values, predictions)

if __name__ == "__main__":
    main()