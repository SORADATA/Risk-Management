# src/var_modules.py
import numpy as np
from scipy.stats import norm

class RiskCalculator:
    def __init__(self, portfolio_metrics):
        self.metrics = portfolio_metrics

    def parametric_var_cvar(self, confidence_level=0.95, days=1):
        """Méthode Variance-Covariance (Hypothèse Normale)"""
        alpha = 1 - confidence_level
        mu, sigma = self.metrics.get_portfolio_performance()
        
        # Ajustement temporel
        horizon_vol = sigma / np.sqrt(252) * np.sqrt(days)
        horizon_ret = mu / 252 * days
        
        # VaR et CVaR
        var = abs(horizon_ret - horizon_vol * norm.ppf(1 - alpha))
        cvar = abs(horizon_ret - (horizon_vol / alpha) * norm.pdf(norm.ppf(alpha)))
        return var, cvar

    def historical_var_cvar(self, confidence_level=0.95):
        """Méthode Historique (Basée sur les données passées réelles)"""
        alpha = 1 - confidence_level
        weighted_returns = self.metrics.get_weighted_returns()
        
        # Percentile direct
        var = -np.percentile(weighted_returns, alpha * 100)
        # Moyenne des pertes pires que la VaR
        cvar = -weighted_returns[weighted_returns <= -var].mean()
        return var, cvar

    def monte_carlo_var_cvar(self, sims=10000, days=1, confidence_level=0.95, stress_factor=1.0):
        """Simulation de Monte Carlo avec option de Stress Test"""
        dt = 1/252
        mu, sigma = self.metrics.get_portfolio_performance()
        
        # Application du Stress Factor (ex: 1.5 pour +50% de volatilité)
        stressed_sigma = sigma * stress_factor
        
        # Génération de scénarios aléatoires
        Z = np.random.normal(0, 1, sims)
        simulated_returns = (mu * dt * days) + (stressed_sigma * np.sqrt(dt * days) * Z)
        
        var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
        cvar = -simulated_returns[simulated_returns <= -var].mean()
        
        return var, cvar, simulated_returns