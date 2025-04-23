import matplotlib.pyplot as plt
import numpy as np

class PortfolioVisualizer:
    @staticmethod
    def plot_cumulative_returns(cumulative_returns, title="Cumulative Returns"):
        plt.figure(figsize=(12, 6))
        cumulative_returns.plot()
        plt.title(title)
        plt.ylabel("Cumulative Returns")
        plt.xlabel("Date")
        plt.grid()
        plt.show()
    
    @staticmethod
    def plot_efficient_frontier(returns, volatilities, sharpe_ratios, optimal_idx):
        plt.figure(figsize=(10, 6))
        plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(volatilities[optimal_idx], returns[optimal_idx], c='red', s=100, marker='*')
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.grid()
        plt.show()
    
    @staticmethod
    def plot_weight_distribution(weights, tickers, title="Portfolio Weights"):
        plt.figure(figsize=(10, 6))
        plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.show()