import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import scipy.optimize as sco
from scipy import stats
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

tickers = ['AAPL', 'MSFT', 'NFLX', 'AMZN', 'GOOG']
start = '2010-01-01'
end = '2020-01-01'
df = pd.DataFrame([yf.download(ticker, start, end)['Adj Close'] for ticker in tickers]).T
df.columns = tickers


def calc_portfolio_perf(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio


def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf):
    results_matrix = np.zeros((len(mean_returns) + 3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio_perf(weights, mean_returns, cov, rf)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_std
        results_matrix[2, i] = sharpe_ratio

        for j in range(len(weights)):
            results_matrix[j + 3, i] = weights[j]

    results_df = pd.DataFrame(results_matrix.T, columns=['ret', 'stdev', 'sharpe'] + [ticker for ticker in tickers])

    return results_df

mean_returns = df.pct_change().mean()
cov = df.pct_change().cov()
num_portfolios = 100000
rf = 0.0
results_frame = simulate_random_portfolios(num_portfolios, mean_returns, cov, rf)

# Indentifica il portafoglio con lo Sharpe Ratio più alto
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
# Identifica il portafoglio con la deviazione standard più piccola
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
# Crea il grafico scatter colorato in base allo Sharpe Ratio
plt.subplots(figsize=(15,10))
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.colorbar()
# Visualizza una stella rossa per il portafoglio con lo Sharpe Ratio più alto
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=500)
# Visualizza una stella verda per il portafoglio con la deviazione standard più piccola
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=500)
plt.show()

print(max_sharpe_port.to_frame().T)

print(min_vol_port.to_frame().T)
