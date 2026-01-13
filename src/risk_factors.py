# src/risk_factors.py
import numpy as np

class PortfolioMetrics:
    def __init__(self, returns, weights):
        self.returns = returns
        self.weights = np.array(weights)
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()

    def get_portfolio_performance(self):
        """Retourne le rendement moyen et la volatilité annualisés du portefeuille."""
        # 252 jours de trading par an
        port_return = np.sum(self.mean_returns * self.weights) * 252
        
        # Formule matricielle de la variance du portefeuille
        port_variance = np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))
        port_volatility = np.sqrt(port_variance) * np.sqrt(252)
        
        return port_return, port_volatility
    
    def get_weighted_returns(self):
        """Retourne la série historique des rendements du portefeuille."""
        return self.returns.dot(self.weights)