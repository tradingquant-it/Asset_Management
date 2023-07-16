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


def calc_neg_sharpe(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return -sharpe_ratio


def max_sharpe_ratio(mean_returns, cov, rf):
    num_assets = len(mean_returns)
    args = (mean_returns, cov, rf)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_neg_sharpe, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


mean_returns = df.pct_change().mean()
cov = df.pct_change().cov()
num_portfolios = 100000
rf = 0.0

optimal_port_sharpe = max_sharpe_ratio(mean_returns, cov, rf)

print(pd.DataFrame([round(x,2) for x in optimal_port_sharpe['x']],index=tickers).T)


def calc_portfolio_std(weights, mean_returns, cov):
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    return portfolio_std

def min_variance(mean_returns, cov):
    num_assets = len(mean_returns)
    args = (mean_returns, cov)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_portfolio_std, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

min_port_variance = min_variance(mean_returns, cov)

print(pd.DataFrame([round(x,2) for x in min_port_variance['x']],index=tickers).T)

