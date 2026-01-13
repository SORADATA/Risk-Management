# main.py
from src.data_loader import DataLoader
from src.risk_factors import PortfolioMetrics
from src.var_modules import RiskCalculator
from src.stress_scenarios import StressTester
import matplotlib.pyplot as plt

def main():
    # 1. CONFIGURATION
    TICKERS = ['AAPL', 'JPM', 'XOM', 'TLT']
    WEIGHTS = [0.4, 0.2, 0.2, 0.2]
    START = "2020-01-01"
    END = "2025-12-31"
    CONFIDENCE = 0.95

    print("--- DÉMARRAGE DU CALCUL DE RISQUE ---")

    # 2. CHARGEMENT DONNÉES
    loader = DataLoader(TICKERS, START, END)
    data = loader.get_data()
    returns = loader.calculate_returns(data)

    # 3. ANALYSE FACTEURS DE RISQUE
    metrics = PortfolioMetrics(returns, WEIGHTS)
    p_ret, p_vol = metrics.get_portfolio_performance()
    print(f"Volatilité annuelle du portefeuille : {p_vol:.2%}")

    # 4. CALCULS VAR/CVAR
    calculator = RiskCalculator(metrics)

    # A. Paramétrique
    var_p, cvar_p = calculator.parametric_var_cvar(CONFIDENCE)
    print(f"\n[Paramétrique] VaR: {var_p:.2%} | CVaR: {cvar_p:.2%}")

    # B. Historique
    var_h, cvar_h = calculator.historical_var_cvar(CONFIDENCE)
    print(f"[Historique]   VaR: {var_h:.2%} | CVaR: {cvar_h:.2%}")

    # C. Monte Carlo & Stress Test
    print("\n--- STRESS TESTING (Monte Carlo) ---")
    scenarios = StressTester.get_scenarios()
    
    results = {}
    
    for name, factor in scenarios.items():
        var, cvar, sims = calculator.monte_carlo_var_cvar(confidence_level=CONFIDENCE, stress_factor=factor)
        results[name] = sims # On garde les simulations pour le graph
        print(f"Scenario {name:<15} (Vol x{factor}) -> VaR: {var:.2%} | CVaR: {cvar:.2%}")

    # 5. VISUALISATION SIMPLE
    # (Tu pourras mettre des visualisations plus complexes dans tes notebooks)
    plt.figure(figsize=(10, 6))
    plt.hist(results['Normal'], bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(results['Crise_Majeure'], bins=50, alpha=0.5, label='Crise Majeure', color='red', density=True)
    plt.title("Distribution des rendements : Normal vs Crise")
    plt.legend()
    plt.show()
    plt.savefig("resultats_simulation.png")
    print("\nGraphique sauvegardé sous 'resultats_simulation.png'")

if __name__ == "__main__":
    main()