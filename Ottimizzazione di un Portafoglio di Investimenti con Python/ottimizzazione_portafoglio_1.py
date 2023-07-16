import yfinance as yf
import numpy as np
import pandas as pd

stock = ['AAPL']
data = yf.download(stock,start='2010-01-01', end='2020-01-01')['Adj Close']
data.sort_index(inplace=True)

returns = data.pct_change()
mean_return = returns.mean()
return_stdev = returns.std()
annualised_return = round(mean_return * 252,2)
annualised_stdev = round(return_stdev * np.sqrt(252),2)
print(f'The annualised mean return of stock {stock[0]} is {annualised_return}, and the annualised volatility is {annualised_stdev}')

# lista titoli in portafoglio
# ATTENZIONE CHE QUESTI DEVONO ESSERE INSERITI IN ORDINE ALFABETICO PER I RISULTATI CORRETTI!!!
stocks = ['AAPL','AMZN','MSFT','FB']
# download dei dati dei prezzi giornalieri per ogni azione nel portafoglio
data = yf.download(stocks, start='2010-01-01', end='2020-01-01')['Adj Close']
data.sort_index(inplace=True)
# converte i prezzi giornalieri in rendimenti giornalieri
returns = data.pct_change()
# calcolo della media e la covarianza dei redimenti giornalieri
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()
# Definizione dei pesi del portafoglio
weights = np.asarray([0.5,0.2,0.2,0.1])
# calcolo del rendimento annualizzato del portafoglio
portfolio_return = round(np.sum(mean_daily_returns * weights) * 252,2)
# calcolo della volatilità annualizzata del portafoglio
portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252),2)
print(f'Portfolio expected annualised return is {portfolio_return} and volatility is {portfolio_std_dev}')

