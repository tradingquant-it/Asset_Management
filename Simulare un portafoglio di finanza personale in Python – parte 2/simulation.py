import pandas as pd
import numpy as np
import random
import yfinance as yf
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

# Imposta il random seed in modo da replicare i risultati
np.random.seed(seed=7)
start, end = "2000-12-31", "2020-01-01"

# download dei dati storici
sp = yf.download("^SP500TR", start=start, end=end)

sp_monthly_pct_return = sp.resample('M').last().pct_change().mean().values[0]
sp_monthly_std_dev = sp.resample('M').last().pct_change().std().values[0]

inflows = {'active_annual_income': 50_000,
           'starting_assets': 250_000}

outflows = {'rent': 1500,
            'credit_card_payment': 750,
            'medical_insurance': 250,
            'pension_contribution': 500,
            'misc': 1500}

variables = {'start_date': "01/01/2020",
             'years': 10,
             'tax_on_active_income_gains': 0.25,
             'avg_ann_income_raise': 0.05,
             'avg_ann_inflation': 0.02,
             'tax_on_investment_gains': 0.35,
             'avg_monthly_market_returns': sp_monthly_pct_return,
             'avg_monthly_market_volatility': sp_monthly_std_dev}
income_gains_storage = []
investment_gains_storage = []

assets_starting_list = [inflows['starting_assets']]
assets_ending_list = []
months = variables['years'] * 12
ruined = False

for month in range(months):

    if assets_ending_list:
        assets_starting_list.append(assets_ending_list[-1])

    assets = assets_starting_list[-1]
    assets -= (outflows['rent'] + outflows['credit_card_payment'] + \
               outflows['medical_insurance'] + outflows['pension_contribution'] + \
               outflows['misc'])

    if assets <= 0:
        inv_gain = 0
        ruined = True
        break
    market_return = np.random.normal(variables['avg_monthly_market_returns'],
                                     variables['avg_monthly_market_volatility'],
                                     1)[0]

    investment_return = (assets * market_return) * (1 - variables['tax_on_investment_gains'])

    investment_gains_storage.append(investment_return)

    assets += investment_return

    income = (inflows['active_annual_income'] * (1 - variables['tax_on_active_income_gains'])) / 12
    income_gains_storage.append(income)

    if (month % 12 == 0):
        inflows['active_annual_income'] *= (1 + (variables['avg_ann_income_raise']))
        outflows['rent'] *= (1 + (variables['avg_ann_inflation']))
        outflows['credit_card_payment'] *= (1 + (variables['avg_ann_inflation']))
        outflows['medical_insurance'] *= (1 + (variables['avg_ann_inflation']))
        outflows['pension_contribution'] *= (1 + (variables['avg_ann_inflation']))
        outflows['misc'] *= (1 + (variables['avg_ann_inflation']))

    assets += income
    assets_ending = assets
    assets_ending_list.append(assets_ending)

plt.plot(pd.Series(assets_ending_list))
plt.xlabel('Month')
plt.ylabel('Ending Asset Value')
plt.show()

#################################################################################################################

np.random.seed(seed=7)
start, end = "2000-12-31", "2020-01-01"
sp = yf.download("^SP500TR", start=start, end=end)

sp_monthly_pct_return = sp.resample('M').last().pct_change().mean().values[0]
sp_monthly_std_dev = sp.resample('M').last().pct_change().std().values[0]

inflows = {'active_annual_income': 50_000,
           'starting_assets': 250_000}

outflows = {'rent': 1500,
            'credit_card_payment': 750,
            'medical_insurance': 250,
            'pension_contribution': 500,
            'misc': 1500}

variables = {'start_date': "01/01/2020",
             'years': 40,
             'retirement_year': 25,
             'tax_on_active_income_gains': 0.25,
             'avg_ann_income_raise': 0.05,
             'avg_ann_inflation': 0.02,
             'tax_on_investment_gains': 0.35,
             'avg_monthly_market_returns': sp_monthly_pct_return,
             'avg_monthly_market_volatility': sp_monthly_std_dev}
income_gains_storage = []
investment_gains_storage = []

assets_starting_list = [inflows['starting_assets']]
assets_ending_list = []
months = variables['years'] * 12

ruined = False

for month in range(months):

    if assets_ending_list:
        assets_starting_list.append(assets_ending_list[-1])

    assets = assets_starting_list[-1]
    assets -= (outflows['rent'] + outflows['credit_card_payment'] + \
               outflows['medical_insurance'] + outflows['pension_contribution'] + \
               outflows['misc'])

    if assets <= 0:
        inv_gain = 0
        ruined = True
        break
    market_return = np.random.normal(variables['avg_monthly_market_returns'],
                                     variables['avg_monthly_market_volatility'],
                                     1)[0]

    investment_return = (assets * market_return) * (1 - variables['tax_on_investment_gains'])

    investment_gains_storage.append(investment_return)

    assets += investment_return

    # Aggiunge la logica per impostare un salario a 0 quando si va in pensione
    if month >= variables['retirement_year'] * 12:
        income = 0
    else:
        income = (inflows['active_annual_income'] * (1 - variables['tax_on_active_income_gains'])) / 12
    income_gains_storage.append(income)

    if (month % 12 == 0):
        inflows['active_annual_income'] *= (1 + (variables['avg_ann_income_raise']))
        outflows['rent'] *= (1 + (variables['avg_ann_inflation']))
        outflows['credit_card_payment'] *= (1 + (variables['avg_ann_inflation']))
        outflows['medical_insurance'] *= (1 + (variables['avg_ann_inflation']))
        outflows['pension_contribution'] *= (1 + (variables['avg_ann_inflation']))
        outflows['misc'] *= (1 + (variables['avg_ann_inflation']))

    assets += income
    assets_ending = assets
    assets_ending_list.append(assets_ending)

plt.plot(pd.Series(assets_ending_list))
plt.xlabel('Month')
plt.ylabel('Ending Asset Value')
plt.show()

##############################################################################################################

np.random.seed(seed=7)
start, end = "2000-12-31", "2020-01-01"
sp = yf.download("^SP500TR", start=start, end=end)

sp_monthly_pct_return = sp.resample('M').last().pct_change().mean().values[0]
sp_monthly_std_dev = sp.resample('M').last().pct_change().std().values[0]

inflows = {'active_annual_income': 50_000,
           'starting_assets': 250_000,
           'monthly_pension': 1500}  # aggiunge l'importo mensile della pensione

outflows = {'rent': 1500,
            'credit_card_payment': 750,
            'medical_insurance': 250,
            'pension_contribution': 500,
            'misc': 1500,
            'retirement_medical_expenses': 850,  # aggiunge la spesa sanitaria dopo la pensione
            'retirement_misc': 2000}  # aggiunge costi generici dopo la pensione

variables = {'start_date': "01/01/2020",
             'years': 40,
             'retirement_year': 25,
             'tax_on_active_income_gains': 0.25,
             'avg_ann_income_raise': 0.05,
             'avg_ann_inflation': 0.02,
             'tax_on_investment_gains': 0.35,
             'avg_monthly_market_returns': sp_monthly_pct_return,
             'avg_monthly_market_volatility': sp_monthly_std_dev}
income_gains_storage = []
investment_gains_storage = []

assets_starting_list = [inflows['starting_assets']]
assets_ending_list = []
months = variables['years'] * 12

ruined = False

for month in range(months):

    if assets_ending_list:
        assets_starting_list.append(assets_ending_list[-1])

    assets = assets_starting_list[-1]

    # Aggiunge la logica per considerare diversi flussi in uscita dopo il pensionamento
    if month >= variables['retirement_year'] * 12:
        outflow = outflows['retirement_medical_expenses'] + outflows['retirement_misc']

    else:

        outflow = (outflows['rent'] + outflows['credit_card_payment'] + \
                   outflows['medical_insurance'] + outflows['pension_contribution'] + \
                   outflows['misc'])

    assets -= outflow

    if assets <= 0:
        inv_gain = 0
        ruined = True
        break
    market_return = np.random.normal(variables['avg_monthly_market_returns'],
                                     variables['avg_monthly_market_volatility'],
                                     1)[0]

    investment_return = (assets * market_return) * (1 - variables['tax_on_investment_gains'])

    investment_gains_storage.append(investment_return)

    assets += investment_return

    # Aggiunge la logica per impostare un salario a 0 quando si va in pensione
    if month >= variables['retirement_year'] * 12:
        income = inflows['monthly_pension']
    else:
        income = (inflows['active_annual_income'] * (1 - variables['tax_on_active_income_gains'])) / 12

    income_gains_storage.append(income)

    if (month % 12 == 0):
        inflows['active_annual_income'] *= (1 + (variables['avg_ann_income_raise']))
        outflows['rent'] *= (1 + (variables['avg_ann_inflation']))
        outflows['credit_card_payment'] *= (1 + (variables['avg_ann_inflation']))
        outflows['medical_insurance'] *= (1 + (variables['avg_ann_inflation']))
        outflows['pension_contribution'] *= (1 + (variables['avg_ann_inflation']))
        outflows['misc'] *= (1 + (variables['avg_ann_inflation']))

    assets += income
    assets_ending = assets
    assets_ending_list.append(assets_ending)

plt.plot(pd.Series(assets_ending_list))
plt.xlabel('Month')
plt.ylabel('Ending Asset Value')
plt.show()


##############################################################################################################

def calculate_outflows(month, outflows, variables):
    # logica per considerare diversi flussi in uscita DOPO il pensionamento
    if month >= variables['retirement_year'] * 12:
        outflow = outflows['retirement_medical_expenses'] + outflows['retirement_misc']

    else:
        # logica per considerare diversi flussi in uscita PRIMA il pensionamento
        outflow = (outflows['rent'] + outflows['credit_card_payment'] + \
                   outflows['medical_insurance'] + outflows['pension_contribution'] + \
                   outflows['misc'])

    # ogni anno incrementa le uscite secondo il tasso d'inflazione
    if (month % 12 == 0) and (month > 0):
        outflows['rent'] *= (1 + (variables['avg_ann_inflation']))
        outflows['credit_card_payment'] *= (1 + (variables['avg_ann_inflation']))
        outflows['medical_insurance'] *= (1 + (variables['avg_ann_inflation']))
        outflows['pension_contribution'] *= (1 + (variables['avg_ann_inflation']))
        outflows['misc'] *= (1 + (variables['avg_ann_inflation']))

    return outflow


def calculate_income(month, inflows, variables):
    # logica per considerare diversi flussi in uscita DOPO il pensionamento
    if month >= variables['retirement_year'] * 12:
        income = inflows['monthly_pension']

    else:
        # logica per considerare diversi flussi in uscita PRIMA il pensionamento
        income = (inflows['active_annual_income'] * (1 - variables['tax_on_active_income_gains'])) / 12
        if (month % 12 == 0) and (month > 0):
            inflows['active_annual_income'] *= (1 + (variables['avg_ann_income_raise']))

    return income


def calculate_investment_gains(assets, variables):
    if assets <= 0:
        inv_gains = 0

    else:
        market_return = np.random.normal(variables['avg_monthly_market_returns'],
                                         variables['avg_monthly_market_volatility'],
                                         1)[0]
        inv_gains = assets * market_return
    return inv_gains


if __name__ == "__main__":
    np.random.seed(seed=7)
    start, end = "2000-12-31", "2020-01-01"
    sp = yf.download("^SP500TR", start=start, end=end)

    sp_monthly_pct_return = sp.resample('M').last().pct_change().mean().values[0]
    sp_monthly_std_dev = sp.resample('M').last().pct_change().std().values[0]

    inflows = {'active_annual_income': 50_000,
               'starting_assets': 250_000,
               'monthly_pension': 1500}

    outflows = {'rent': 1500,
                'credit_card_payment': 750,
                'medical_insurance': 250,
                'pension_contribution': 500,
                'misc': 1500,
                'retirement_medical_expenses': 850,
                'retirement_misc': 2000}

    variables = {'start_date': "01/01/2020",
                 'years': 40,
                 'retirement_year': 25,
                 'tax_on_active_income_gains': 0.25,
                 'avg_ann_income_raise': 0.05,
                 'avg_ann_inflation': 0.02,
                 'tax_on_investment_gains': 0.35,
                 'avg_monthly_market_returns': sp_monthly_pct_return,
                 'avg_monthly_market_volatility': sp_monthly_std_dev}
    income_gains_storage = []
    investment_gains_storage = []

    assets_starting_list = [inflows['starting_assets']]
    assets_ending_list = []
    months = variables['years'] * 12

    ruined = False

    for month in range(months):

        if assets_ending_list:
            assets_starting_list.append(assets_ending_list[-1])

        assets = assets_starting_list[-1]

        # calola le uscite tramite la funzione
        outflow = calculate_outflows(month, outflows, variables)

        assets -= outflow

        # Modificato il blocco "if" per includere la nuova funzione
        if assets <= 0:
            ruined = True
            break

        # usa la funzione per calcolare i rendimenti dell'investimento
        investment_return = calculate_investment_gains(assets, variables)

        investment_gains_storage.append(investment_return)

        assets += investment_return

        # calcola le entrate tramite la funzione
        income = calculate_income(month, inflows, variables)

        income_gains_storage.append(income)

        assets += income
        assets_ending = assets
        assets_ending_list.append(assets_ending)

    plt.plot(pd.Series(assets_ending_list))
    plt.xlabel('Month')
    plt.ylabel('Ending Asset Value')
    plt.show()
