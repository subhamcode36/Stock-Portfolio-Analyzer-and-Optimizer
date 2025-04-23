import yfinance as yf
import pandas as pd
import os

class StockDataFetcher:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_data(self, tickers, start_date, end_date, save_to_file=True):
        """
        Fetch historical stock data from Yahoo Finance
        """
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        if save_to_file:
            file_path = os.path.join(self.data_dir, "stock_data.csv")
            data.to_csv(file_path)
            
        return data
    
    def load_data_from_file(self):
        """
        Load previously saved stock data
        """
        file_path = os.path.join(self.data_dir, "stock_data.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError("No saved data found. Fetch data first.")
            
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    def calculate_returns(self, data):
        """
        Calculate daily and cumulative returns
        """
        if data.empty:
            raise ValueError("No data available. Fetch data first.")
            
        daily_returns = data.pct_change().dropna()
        cumulative_returns = (1 + daily_returns).cumprod()
        
        return daily_returns, cumulative_returns