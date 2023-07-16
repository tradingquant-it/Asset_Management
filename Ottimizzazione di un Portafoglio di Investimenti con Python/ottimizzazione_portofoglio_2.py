import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Lista dei titoli in portafoglio
stocks = ['AAPL', 'AMZN', 'MSFT', 'FB']
# Download dei dati dei prezzi giornalieri per ogni azione nel portafoglio
data = yf.download(stocks, start='2010-01-01', end='2020-01-01')['Adj Close']
data.sort_index(inplace=True)
# Converte i prezzi giornalieri in rendimenti giornalieri
returns = data.pct_change()
# Calcolo della media e la covarianza dei redimenti giornalieri
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

# Imposta il numero di simulazioni
num_portfolios = 25000
# Imposta un array dei risultati
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    # Selezioni di pesi random
    weights = np.random.random(4)
    # Ribilanciamento dei pesi per avere somma 1
    weights /= np.sum(weights)

    # Calcolo dei rendimenti e volatilità del portafoglio
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    # Memorizzazioni nell'array dei risultati
    results[0, i] = portfolio_return
    results[1, i] = portfolio_std_dev
    # Memorizzazione dello Sharpe Ratio - elemento risk free escluso per semplicità
    results[2, i] = results[0, i] / results[1, i]

# Converi l'array dei risultati in un dataframe pandas
results_frame = pd.DataFrame(results.T, columns=['ret', 'stdev', 'sharpe'])
# Crea un grafico scatter colorato dallo Sharpe Ratio
plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.sharpe, cmap='RdYlBu')
plt.colorbar()
plt.show()
print("")