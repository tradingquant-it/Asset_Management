import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Lista dei titoli in portafoglio
stocks = ['AAPL', 'AMZN', 'MSFT', 'META']
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
results = np.zeros((7, num_portfolios))

for i in range(num_portfolios):
    # Selezioni di pesi random
    weights = np.array(np.random.random(4))

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

    # ciclo sul vettore dei pesi e aggiunta dei dati sull'array dei risultati
    for j in range(len(weights)):
        results[j + 3, i] = weights[j]

# Converte l'array dei risultati in un dataframe pandas
results_frame = pd.DataFrame(results.T, columns=['ret', 'stdev', 'sharpe', stocks[0], stocks[1], stocks[2], stocks[3]])
# Individua il portafoglio con il Sharpe Ratio maggiore
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
# Individua il portafoglio con la deviazione standard minima
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
# Crea il grafico scatter colorato dallo Sharpe Ratio
plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.sharpe, cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()
# Visualizza una stella rossa per evidenziare il portafoglio con lo Sharpe Ratio maggiore
plt.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(5, 1, 0), color='r', s=1000)
# Visualizza una stella verde per evidenziare il portafoglio con la varianza minima
plt.scatter(min_vol_port[1], min_vol_port[0], marker=(5, 1, 0), color='g', s=1000)

plt.show()
print("")

print(max_sharpe_port)

print(min_vol_port)