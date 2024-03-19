import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.

    Parameters:
    - ticker: str, the ticker symbol of the stock (e.g., 'AAPL' for Apple Inc.).
    - start_date: str, start date in 'YYYY-MM-DD' format.
    - end_date: str, end date in 'YYYY-MM-DD' format.

    Returns:
    - df: DataFrame, the stock data.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    # Optionally, reset the index to make 'Date' a column
    df.reset_index(inplace=True)
    return df

def preprocess_data(df):
    """
    Prepares stock data for model training.

    Parameters:
    - df: DataFrame, the stock data with 'Open' as a feature and 'Close' as the target.

    Returns:
    - X: np.array, the feature data.
    - y: np.array, the target data.
    """
    df['Open'] = df['Open'].fillna(method='ffill')

    X = df[['Open']].values
    y = df['Close'].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2021-01-01'

    # Fetch and preprocess data
    df = fetch_stock_data(ticker, start_date, end_date)
    X, y = preprocess_data(df)
    input_shape = (X.shape[1], 1)  # Assuming 'Open' is the only feature

    # Build and train the model
    model = build_model(input_shape)
    # For demonstration, using a simple split; consider a more appropriate method for time series
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
