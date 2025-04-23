import numpy as np
import pandas as pd

class PortfolioAnalyzer:
    def __init__(self, weights=None, risk_free_rate=0.0):
        self.weights = weights
        self.risk_free_rate = risk_free_rate
    
    def set_weights(self, weights):
        """Set portfolio weights"""
        if not np.isclose(sum(weights), 1):
            raise ValueError("Weights must sum to 1")
        self.weights = np.array(weights)
    
    def portfolio_performance(self, daily_returns):
        """
        Calculate portfolio returns and volatility
        """
        if self.weights is None:
            raise ValueError("Weights not set")
            
        # Annualized portfolio return
        port_return = np.sum(daily_returns.mean() * self.weights) * 252
        
        # Annualized portfolio volatility
        port_volatility = np.sqrt(np.dot(self.weights.T, 
                                       np.dot(daily_returns.cov() * 252, 
                                              self.weights)))
        
        # Sharpe ratio
        sharpe_ratio = (port_return - self.risk_free_rate) / port_volatility
        
        return {
            'return': port_return,
            'volatility': port_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def calculate_var(self, daily_returns, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR)
        """
        portfolio_returns = daily_returns.dot(self.weights)
        return np.percentile(portfolio_returns, 100 * (1 - confidence_level))
    
    def calculate_beta(self, daily_returns, market_returns):
        """
        Calculate beta for each stock against market returns
        """
        cov_matrix = np.cov(daily_returns.T, market_returns)
        market_variance = np.var(market_returns)
        
        betas = cov_matrix[-1, :-1] / market_variance
        portfolio_beta = np.dot(self.weights, betas)
        
        return portfolio_beta, betas