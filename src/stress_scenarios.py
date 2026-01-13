# src/stress_scenarios.py

class StressTester:
    @staticmethod
    def get_scenarios():
        """Définit différents scénarios de crise."""
        return {
            "Normal": 1.0,           # Volatilité normale
            "Crise_Moderee": 1.25,   # +25% Volatilité
            "Crise_Majeure": 1.50,   # +50% Volatilité (Type 2008)
            "Black_Swan": 2.0        # +100% Volatilité (Type COVID mars 2020)
        }