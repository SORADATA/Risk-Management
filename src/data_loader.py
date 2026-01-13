# src/data_loader.py
import yfinance as yf
import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start = start_date
        self.end = end_date

    def get_data(self):
        """Télécharge les données de clôture ajustées."""
        print(f"Chargement des données pour {self.tickers}...")
        data = yf.download(self.tickers, start=self.start, end=self.end)['Close']
        return data

    def calculate_returns(self, data):
        """Calcule les rendements logarithmiques."""
        # Log returns sont préférables pour l'additivité temporelle
        returns = np.log(data / data.shift(1)).dropna()
        return returns