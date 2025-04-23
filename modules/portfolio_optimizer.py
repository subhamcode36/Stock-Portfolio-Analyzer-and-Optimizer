import numpy as np
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, daily_returns, risk_free_rate=0.0):
        self.daily_returns = daily_returns
        self.num_assets = daily_returns.shape[1]
        self.risk_free_rate = risk_free_rate
    
    def negative_sharpe_ratio(self, weights):
        """Negative Sharpe ratio for minimization"""
        port_return = np.sum(self.daily_returns.mean() * weights) * 252
        port_volatility = np.sqrt(np.dot(weights.T, 
                                       np.dot(self.daily_returns.cov() * 252, 
                                              weights)))
        return -(port_return - self.risk_free_rate) / port_volatility
    
    def portfolio_variance(self, weights):
        """Calculate portfolio variance for minimum variance optimization"""
        return np.dot(weights.T, np.dot(self.daily_returns.cov() * 252, weights))
    
    def optimize_portfolio(self, optimization_type='sharpe'):
        """Optimize portfolio based on specified objective"""
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = [1./self.num_assets] * self.num_assets
        
        if optimization_type == 'sharpe':
            objective = self.negative_sharpe_ratio
        elif optimization_type == 'min_variance':
            objective = self.portfolio_variance
        else:
            raise ValueError("Invalid optimization type. Use 'sharpe' or 'min_variance'")
        
        opt_result = minimize(objective,
                            initial_guess,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
        
        return opt_result.x