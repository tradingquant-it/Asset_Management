import numpy as np
import pandas as pd
from numpy.linalg import inv

asset_returns_orig = pd.read_csv('rendimenti_asset.csv', index_col='Year', parse_dates=True)
asset_weights = pd.read_csv('pesi_asset.csv', index_col='asset_class')
cols = ['Global Bonds (Unhedged)', 'Total US Bond Market', 'US Large Cap Growth',
        'US Large Cap Value', 'US Small Cap Growth', 'US Small Cap Value', 'Emerging Markets',
        'Intl Developed ex-US Market', 'Short Term Treasury']
asset_returns = asset_returns_orig[cols].dropna()
treasury_rate = asset_returns['Short Term Treasury']
asset_returns = asset_returns[cols[:-1]].astype(float).dropna()
asset_weights = asset_weights.loc[cols[:-1]]

print(asset_returns.mean())

print(asset_weights)

excess_asset_returns = asset_returns.subtract(treasury_rate, axis=0)
cov = excess_asset_returns.cov()
global_return = excess_asset_returns.mean().multiply(asset_weights['weight'].values).sum()
market_var = np.matmul(asset_weights.values.reshape(len(asset_weights)).T,
                                       np.matmul(cov.values, asset_weights.values.reshape(len(asset_weights))))
print(f'The global market mean return is {global_return:.4f} and the variance is {market_var:.6}')
risk_aversion = global_return / market_var
print(f'The risk aversion parameter is {risk_aversion:.2f}')


def implied_rets(risk_aversion, sigma, w):
    implied_rets = risk_aversion * sigma.dot(w).squeeze()
    return implied_rets

implied_equilibrium_returns = implied_rets(risk_aversion, cov, asset_weights)
print(implied_equilibrium_returns)

Q = np.array([0.0925, 0.005, 0.055])

P = np.asarray([[0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, .85, -.85, .15, -.15, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]])

view1_var = np.matmul(P[0].reshape(len(P[0])),np.matmul(cov.values, P[0].reshape(len(P[0])).T))
view2_var = np.matmul(P[1].reshape(len(P[1])),np.matmul(cov.values, P[1].reshape(len(P[1])).T))
view3_var = np.matmul(P[2].reshape(len(P[2])),np.matmul(cov.values, P[2].reshape(len(P[2])).T))
print(f'The Variance of View 1 Portfolio is {view1_var}, and the standard deviation is {np.sqrt(view1_var):.3f}\n',\
      f'The Variance of View 2 Portfolio is {view2_var}, and the standard deviation is {np.sqrt(view2_var):.3f}\n',\
      f'The Variance of View 3 Portfolio is {view3_var}, and the standard deviation is {np.sqrt(view3_var):.3f}')


def error_cov_matrix(sigma, tau, P):
    matrix = np.diag(np.diag(P.dot(tau * cov).dot(P.T)))
    return matrix
tau = 0.025
omega = error_cov_matrix(cov, tau, P)

sigma_scaled = cov * tau
BL_return_vector = implied_equilibrium_returns + sigma_scaled.dot(P.T).dot(inv(P.dot(sigma_scaled).dot(P.T) + omega).dot(Q - P.dot(implied_equilibrium_returns)))

print(BL_return_vector)

returns_table = pd.concat([implied_equilibrium_returns, BL_return_vector], axis=1) * 100
returns_table.columns = ['Implied Returns', 'BL Return Vector']
returns_table['Difference'] = returns_table['BL Return Vector'] - returns_table['Implied Returns']
returns_table.style.format('{:,.2f}%')

inverse_cov = pd.DataFrame(inv(cov.values), index=cov.columns, columns=cov.index)
BL_weights_vector = inverse_cov.dot(BL_return_vector)
BL_weights_vector = BL_weights_vector/sum(BL_weights_vector)

# Calcola i pesi ottimizzati media-varianza
MV_weights_vector = inverse_cov.dot(excess_asset_returns.mean())
MV_weights_vector = MV_weights_vector/sum(MV_weights_vector)
weights_table = pd.concat([BL_weights_vector, asset_weights, MV_weights_vector], axis=1) * 100
weights_table.columns = ['BL Weights', 'Market Cap Weights', 'Mean-Var Weights']
weights_table['BL/Mkt Cap Diff'] = weights_table['BL Weights'] - weights_table['Market Cap Weights']
weights_table.style.format('{:,.2f}%')

import matplotlib.pyplot as plt
N = BL_weights_vector.shape[0]
fig, ax = plt.subplots(figsize=(15, 7))
ax.set_title('Black-Litterman Model Portfolio Weights Recommendation vs the Market Portfolio vs Mean-Variance Weights')
ax.plot(np.arange(N)+1, MV_weights_vector, '^', c='b', label='Mean-Variance)')
ax.plot(np.arange(N)+1, asset_weights, 'o', c='g', label='Market Portfolio)')
ax.plot(np.arange(N)+1, BL_weights_vector, '*', c='r',markersize=10, label='Black-Litterman')
ax.vlines(np.arange(N)+1, 0, BL_weights_vector, lw=1)
ax.vlines(np.arange(N)+1, 0, MV_weights_vector, lw=1)
ax.vlines(np.arange(N)+1, 0, asset_weights, lw=1)
ax.axhline(0, c='m')
ax.axhline(-1, c='m', ls='--')
ax.axhline(1, c='m', ls='--')
ax.set_xlabel('Assets')
ax.set_xlabel('Portfolio Weighting')
ax.xaxis.set_ticks(np.arange(1, N+1, 1))
ax.set_xticklabels(asset_weights.index.values)
plt.xticks(rotation=90, )
plt.legend(numpoints=1, fontsize=11)
plt.show()

from mlfinlab.portfolio_optimization.bayesian import VanillaBlackLitterman

views = [0.0925, 0.005, 0.055]
pick_list = [
    {"Emerging Markets": 1.0},
    {"US Large Cap Growth": 0.85,
     "US Large Cap Value": -0.85,
     "US Small Cap Growth": 0.15,
     "US Small Cap Value": -0.15},
    {"Intl Developed ex-US Market": 1.0}]

bl = VanillaBlackLitterman()
bl.allocate(covariance=cov,
            market_capitalised_weights=asset_weights,
            investor_views=views,
            pick_list=pick_list,
            asset_names=cov.columns,
            tau=tau,
            risk_aversion=risk_aversion)

print(bl.implied_equilibrium_returns.T)

print(bl.posterior_expected_returns.T)

weights_table2 = pd.concat([bl.weights.T[0], BL_weights_vector], axis=1) * 100
weights_table2.columns = ['mlfinlab', 'Initial Results']
weights_table2['Difference'] = weights_table2['Initial Results'] - weights_table2['mlfinlab']
weights_table2.style.format('{:,.2f}%')

