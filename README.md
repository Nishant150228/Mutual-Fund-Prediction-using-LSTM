# Stock Price Prediction using LSTM

This project demonstrates the use of Long Short-Term Memory (LSTM) neural networks for predicting future stock prices based on historical stock data. The model is implemented in Python using TensorFlow and leverages Yahoo Finance data for training and prediction.

---

## Key Features

1. **Data Collection**: Historical stock price data is fetched from Yahoo Finance using the `yfinance` library.
2. **Data Preprocessing**: Data is scaled between 0 and 1 using MinMaxScaler for improved model performance.
3. **LSTM Architecture**: 
   - Two LSTM layers with 100 units each.
   - Dropout layers to prevent overfitting.
   - Fully connected dense layers for final predictions.
4. **Prediction**: Predicts stock prices for the entire year (365 days) using the trained model.
5. **Visualization**: Interactive Plotly graphs visualize historical and predicted stock prices.

---

## Installation

### Requirements

- Python 3.7+
- Libraries:
  - `numpy`
  - `pandas`
  - `yfinance`
  - `scikit-learn`
  - `tensorflow`
  - `plotly`

Install the required libraries using pip:
```bash
pip install numpy pandas yfinance scikit-learn tensorflow plotly
```

---

## Usage

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Run the Python script:
   ```bash
   python stock_price_prediction.py
   ```
3. Enter the stock ticker symbol and date range when prompted.
4. The model will predict stock prices for the year 2024 and display an interactive graph.

---

## Example

For Hindustan Petroleum Corporation Limited (HINDPETRO.NS):

- **Input**:
  - Ticker: `HINDPETRO.NS`
  - Date range: January 2021 to December 2023

- **Output**:
  - Historical stock price trends (blue line).
  - Predicted stock prices for 2024 (red line).

---

## Model Details

- **Architecture**:
  - LSTM layers: 100 units with `relu` activation.
  - Dropout: 20% to reduce overfitting.
  - Dense layers for final output.
- **Training**:
  - Optimizer: `adam`
  - Loss function: Mean Squared Error (MSE)
  - Epochs: 50
  - Batch size: 32

---

## Results

The model captures historical trends effectively and provides a forecast for the next year. However, care should be taken to validate predictions, as models can overfit or misinterpret market dynamics.

---

## Visualization

The graph includes:

- **Historical Prices**: Stock prices from 2021 to 2023.
- **Predicted Prices**: Forecast for 2024.

---

## Future Work

1. Enhance the model by incorporating:
   - Volume data.
   - External features like market sentiment or news events.
2. Perform hyperparameter tuning to improve prediction accuracy.
3. Validate predictions using additional datasets.
4. Implement real-time prediction capabilities.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing historical stock data.
- TensorFlow and Plotly communities for open-source tools and libraries.

---
