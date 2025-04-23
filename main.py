from modules.data_fetcher import StockDataFetcher
from modules.portfolio_analyzer import PortfolioAnalyzer
from modules.portfolio_optimizer import PortfolioOptimizer
from utils.visualizer import PortfolioVisualizer
from datetime import datetime, timedelta
import numpy as np

class StockPortfolioApp:
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
        self.analyzer = PortfolioAnalyzer(risk_free_rate=0.02)  # 2% risk-free rate
        self.optimizer = None
        self.visualizer = PortfolioVisualizer()
    
    def get_user_input(self):
        print("\nStock Portfolio Analyzer and Optimizer")
        print("------------------------------------")
        
        tickers = input("Enter stock tickers (comma separated, e.g., AAPL,MSFT,GOOG): ").strip().upper().split(',')
        tickers = [t.strip() for t in tickers if t.strip()]
        
        years = float(input("Enter number of years of historical data to fetch (e.g., 5): "))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        return tickers, start_date, end_date
    
    def run_analysis(self, tickers, start_date, end_date):
        print("\nFetching data...")
        try:
            # Try to load saved data first
            data = self.data_fetcher.load_data_from_file()
            print("Loaded saved data")
        except FileNotFoundError:
            # Fetch fresh data if no saved data exists
            data = self.data_fetcher.fetch_data(tickers, start_date, end_date)
            print("Fetched new data from Yahoo Finance")
        
        daily_returns, cumulative_returns = self.data_fetcher.calculate_returns(data)
        
        # Initialize optimizer
        self.optimizer = PortfolioOptimizer(daily_returns, risk_free_rate=0.02)
        
        # Current portfolio analysis (equal weights)
        weights = np.array([1/len(tickers)] * len(tickers))
        self.analyzer.set_weights(weights)
        perf = self.analyzer.portfolio_performance(daily_returns)
        
        print("\nCurrent Portfolio (Equal Weights):")
        print(f"Annual Return: {perf['return']*100:.2f}%")
        print(f"Annual Volatility: {perf['volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        
        # Calculate VaR
        var = self.analyzer.calculate_var(daily_returns)
        print(f"Daily Value at Risk (95% confidence): {var*100:.2f}%")
        
        # Optimized portfolio (max Sharpe ratio)
        opt_weights = self.optimizer.optimize_portfolio(optimization_type='sharpe')
        self.analyzer.set_weights(opt_weights)
        opt_perf = self.analyzer.portfolio_performance(daily_returns)
        
        print("\nOptimized Portfolio (Max Sharpe Ratio):")
        for ticker, weight in zip(tickers, opt_weights):
            print(f"{ticker}: {weight*100:.2f}%")
        print(f"\nAnnual Return: {opt_perf['return']*100:.2f}%")
        print(f"Annual Volatility: {opt_perf['volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {opt_perf['sharpe_ratio']:.2f}")
        
        # Minimum variance portfolio
        min_var_weights = self.optimizer.optimize_portfolio(optimization_type='min_variance')
        self.analyzer.set_weights(min_var_weights)
        min_var_perf = self.analyzer.portfolio_performance(daily_returns)
        
        print("\nMinimum Variance Portfolio:")
        for ticker, weight in zip(tickers, min_var_weights):
            print(f"{ticker}: {weight*100:.2f}%")
        print(f"\nAnnual Return: {min_var_perf['return']*100:.2f}%")
        print(f"Annual Volatility: {min_var_perf['volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {min_var_perf['sharpe_ratio']:.2f}")
        
        # Visualization
        self.visualizer.plot_cumulative_returns(cumulative_returns)
        self.visualizer.plot_weight_distribution(weights, tickers, "Equal Weights Portfolio")
        self.visualizer.plot_weight_distribution(opt_weights, tickers, "Optimized Portfolio (Max Sharpe)")
        self.visualizer.plot_weight_distribution(min_var_weights, tickers, "Minimum Variance Portfolio")
        
        # Generate efficient frontier
        self.generate_efficient_frontier(daily_returns, tickers)
    
    def generate_efficient_frontier(self, daily_returns, tickers):
        print("\nGenerating efficient frontier...")
        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        
        for i in range(num_portfolios):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            
            self.analyzer.set_weights(weights)
            perf = self.analyzer.portfolio_performance(daily_returns)
            
            results[0,i] = perf['return']
            results[1,i] = perf['volatility']
            results[2,i] = perf['sharpe_ratio']
        
        optimal_idx = np.argmax(results[2])
        
        self.visualizer.plot_efficient_frontier(
            returns=results[0,:],
            volatilities=results[1,:],
            sharpe_ratios=results[2,:],
            optimal_idx=optimal_idx
        )

if __name__ == "__main__":
    app = StockPortfolioApp()
    tickers, start_date, end_date = app.get_user_input()
    app.run_analysis(tickers, start_date, end_date)