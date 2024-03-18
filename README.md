# Neural-Network-Stock-Prediction
Stock Price Prediction using LSTM Neural Network
This project is a Python implementation of a Long Short-Term Memory (LSTM) neural network for predicting stock prices. The model uses historical stock data, including closing prices, trading volume, and technical indicators (Simple Moving Average), to forecast future stock prices.

Features
Downloads historical stock data from Yahoo Finance
Preprocesses the data by scaling and creating training/testing datasets
Builds and trains an LSTM neural network model
Predicts future stock prices based on the trained model
Visualizes the predicted prices along with the actual historical prices
Supports command-line arguments for specifying stock symbol, start date, and end date
Includes a batch script (run.bat) for automatic installation and execution on Windows
Checks for CUDA and cuDNN installations to enable GPU acceleration (if available)
Installs and launches an Electron application for a graphical user interface (GUI)
Installation
Clone the repository:
git clone https://github.com/your-username/stock-prediction.git

Navigate to the project directory:
cd stock-prediction

Run the run.bat script:
run.bat

This script will handle the following tasks:

Check for Python and Node.js installations, and install Node.js if not found
Create a virtual environment and install required Python packages
Check for CUDA and cuDNN installations, and configure TensorFlow to use GPU acceleration if available
Install Electron and front-end dependencies
Run the Python script for stock prediction
Launch the Electron application with a graphical user interface
Note: The script assumes you have Python 3 installed on your system. If not, you'll need to install Python 3 manually before running the script.

Usage
After running the run.bat script, the Electron application will launch, providing a graphical user interface for interacting with the stock prediction model.

Alternatively, you can run the Python script directly from the command line:

python stock_prediction.py --symbol=AAPL --start_date=2020-01-01 --end_date=2023-01-01

Replace AAPL with the stock symbol of your choice, and adjust the start_date and end_date parameters as needed.

Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

License
This project is licensed under the MIT License.
