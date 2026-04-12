# BTC Price Direction Classifier 📈

Predicts whether Bitcoin will be higher 5 days from now using
XGBoost trained on technical indicators. Includes a simple
long/short backtest evaluated with the Sharpe ratio.

## Results
| Split      | Precision | Recall | F1   |
|------------|-----------|--------|------|
| CV Mean    | X.XX      | X.XX   | X.XX |
| Validation | X.XX      | X.XX   | X.XX |
| Test       | X.XX      | X.XX   | X.XX |

Backtest Sharpe Ratio: X.XX

## Quickstart
pip install -r requirements.txt
python main.py

## Stack
Python · XGBoost · LightGBM · scikit-learn · yfinance · ta