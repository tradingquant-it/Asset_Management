import pandas as pd
import numpy as np
import random
import yfinance as yf
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,6)

# Importa il modello dei flussi
inflows = {'active_annual_income':50_000}
variables = {'start_date' : "01/01/2020",
             'years': 10}
income_gains_storage = []
months = variables['years'] * 12

for month in range(months):
    income = inflows['active_annual_income'] / 12
    income_gains_storage.append(income)
plt.plot(pd.Series(income_gains_storage).cumsum())
plt.show()

inflows = {'active_annual_income': 50_000}
variables = {'start_date': "01/01/2020",
             'years': 10,
             'tax_on_active_income_gains': 0.25}  # Aggiunge la % di tasse sui redditi attivi
income_gains_storage = []
months = variables['years'] * 12

for month in range(months):
    # Aggiunge l'effetto delle tasse applicate ai redditi
    income = (inflows['active_annual_income'] * (1 - variables['tax_on_active_income_gains'])) / 12
    income_gains_storage.append(income)

plt.plot(pd.Series(income_gains_storage).cumsum())
plt.xlabel('Month')
plt.ylabel('Cumulative Income')
plt.show()

###########################################################################################################

inflows = {'active_annual_income': 50_000}
variables = {'start_date': "01/01/2020",
             'years': 10,
             'tax_on_active_income_gains': 0.25,
             'avg_ann_income_raise': 0.05} # Aggiunge la % di tasse sui redditi attivi
income_gains_storage = []
months = variables['years'] * 12

for month in range(months):
    income = (inflows['active_annual_income'] * (1 - variables['tax_on_active_income_gains'])) / 12
    income_gains_storage.append(income)
    # ogni 12 mesi aumenta il salario base di una percentuale media annuale
    if (month % 12 == 0) and (month > 0):  # non applichiamo l'aumento al primo mese
        inflows['active_annual_income'] *= (1 + (variables['avg_ann_income_raise']))

plt.plot(pd.Series(income_gains_storage).cumsum())
plt.xlabel('Month')
plt.ylabel('Cumulative Income')
plt.show()

###########################################################################################################

# Scaricare i dati storici dei prezzi dell'S&P500
start, end = "2000-12-31", "2020-01-01"
tickers = ["^SP500TR"]
sp = pd.DataFrame([yf.download(ticker, start, end).loc[:, 'Adj Close'] for ticker in tickers],
                  index=tickers).T.fillna(method='ffill')
# Calcolare i rendimenti e volatità medie mensili
sp_monthly_pct_return = sp.resample('M').last().pct_change().mean().values[0]
sp_monthly_std_dev = sp.resample('M').last().pct_change().std().values[0]

inflows = {'active_annual_income': 50_000,
           'starting_assets': 250_000}  # aggiungere il valore iniziale degli asset
variables = {'start_date': "01/01/2020",
             'years': 10,
             'tax_on_active_income_gains': 0.25,
             'avg_ann_income_raise': 0.05,
             'avg_monthly_market_returns': sp_monthly_pct_return,  # aggiunge i dati dei rendimento del mercato
             'avg_monthly_market_volatility': sp_monthly_std_dev}  # aggiunge i dati della volatilità del mercato
income_gains_storage = []
investment_gains_storage = []  # crea la lista per memorizzare i rendimenti degli investimenti
# crea la lista per memorizzare il valore degli asset all'inizio del periodo
assets_starting_list = [inflows['starting_assets']]  # imposta il primo valore della lista come asset del giorno
assets_ending_list = []  # crea la lista per memorizzare il valore degli asset alla fine del periodo
months = variables['years'] * 12

for month in range(months):

    # Verifica se è la prima volta che eseguiamo il ciclo mensile e, in caso contrario,
    # usiamo gli asset finali del periodo precedente come asset iniziali di questo periodo.
    if assets_ending_list:
        assets_starting_list.append(assets_ending_list[-1])

    income = (inflows['active_annual_income'] * (1 - variables['tax_on_active_income_gains'])) / 12
    income_gains_storage.append(income)

    if (month % 12 == 0) and (month > 0):
        inflows['active_annual_income'] *= (1 + (variables['avg_ann_income_raise']))

    # genera random un rendimento mensile del mercato da una distribuzione normale
    market_return = np.random.normal(variables['avg_monthly_market_returns'],
                                     variables['avg_monthly_market_volatility'],
                                     1)[0]

    # calcola il rendimento dell'investimento
    investment_return = assets_starting_list[-1] * market_return
    # memorizza in una lista il valore del rendimento dell'investimento
    investment_gains_storage.append(investment_return)

    # calcola il valore degli asset alla fine del periodo
    assets_ending = assets_starting_list[-1] + investment_return + income
    # memorizza il valore finale degli asset
    assets_ending_list.append(assets_ending)

plt.plot(pd.Series(investment_gains_storage).cumsum())
plt.xlabel('Month')
plt.ylabel('Cumulative Investment Returns')
plt.show()

plt.plot(pd.Series(assets_ending_list))
plt.xlabel('Month')
plt.ylabel('Ending Asset Value')
plt.show()

###########################################################################################################

start, end = "2000-12-31", "2020-01-01"
tickers = ["^SP500TR"]
sp = pd.DataFrame([yf.download(ticker, start, end).loc[:, 'Adj Close'] for ticker in tickers],
                  index=tickers).T.fillna(method='ffill')

sp_monthly_pct_return = sp.resample('M').last().pct_change().mean().values[0]
sp_monthly_std_dev = sp.resample('M').last().pct_change().std().values[0]

inflows = {'active_annual_income': 50_000,
           'starting_assets': 250_000}
variables = {'start_date': "01/01/2020",
             'years': 10,
             'tax_on_active_income_gains': 0.25,
             'avg_ann_income_raise': 0.05,
             'tax_on_investment_gains': 0.35,  # aggiugere la % di tasse seui rendimenti degli investimenti
             'avg_monthly_market_returns': sp_monthly_pct_return,
             'avg_monthly_market_volatility': sp_monthly_std_dev}
income_gains_storage = []
investment_gains_storage = []

assets_starting_list = [inflows['starting_assets']]
assets_ending_list = []
months = variables['years'] * 12

for month in range(months):

    if assets_ending_list:
        assets_starting_list.append(assets_ending_list[-1])

    income = (inflows['active_annual_income'] * (1 - variables['tax_on_active_income_gains'])) / 12
    income_gains_storage.append(income)

    if (month % 12 == 0) and (month > 0):
        inflows['active_annual_income'] *= (1 + (variables['avg_ann_income_raise']))

    market_return = np.random.normal(variables['avg_monthly_market_returns'],
                                     variables['avg_monthly_market_volatility'],
                                     1)[0]

    # applica la % di tasse ai guadagni degli investimenti
    investment_return = (assets_starting_list[-1] * market_return) * (1 - variables['tax_on_investment_gains'])
    # memorizza in una lista il valore dei rendimenti dell'investimento
    investment_gains_storage.append(investment_return)

    # calcola il valore degli asset alla fine del periodo
    assets_ending = assets_starting_list[-1] + investment_return + income
    # memorizza il valore finale degli asset
    assets_ending_list.append(assets_ending)

plt.plot(pd.Series(investment_gains_storage).cumsum())
plt.xlabel('Month')
plt.ylabel('Cumulative Investment Returns')
plt.show()

#################################################################################################################

start, end = "2000-12-31", "2020-01-01"
tickers = ["^SP500TR"]
sp = pd.DataFrame([yf.download(ticker, start, end).loc[:, 'Adj Close'] for ticker in tickers],
                  index=tickers).T.fillna(method='ffill')

sp_monthly_pct_return = sp.resample('M').last().pct_change().mean().values[0]
sp_monthly_std_dev = sp.resample('M').last().pct_change().std().values[0]

inflows = {'active_annual_income': 50_000,
           'starting_assets': 250_000}
# aggiungere il dizionario outflows
outflows = {'rent': 1500,
            'credit_card_payment': 750,
            'medical_insurance': 250,
            'pension_contribution': 500,
            'misc': 1500}

variables = {'start_date': "01/01/2020",
             'years': 10,
             'tax_on_active_income_gains': 0.25,
             'avg_ann_income_raise': 0.05,
             'avg_ann_inflation': 0.02,  # add annual inflation rate
             'tax_on_investment_gains': 0.35,
             'avg_monthly_market_returns': sp_monthly_pct_return,
             'avg_monthly_market_volatility': sp_monthly_std_dev}
income_gains_storage = []
investment_gains_storage = []

assets_starting_list = [inflows['starting_assets']]
assets_ending_list = []
months = variables['years'] * 12

for month in range(months):

    if assets_ending_list:
        assets_starting_list.append(assets_ending_list[-1])

    # crea una varibiale per memorizzare il valore degli asset
    assets = assets_starting_list[-1]
    # sottrae il costo mensile dal capitale degli asset
    assets -= (outflows['rent'] + outflows['credit_card_payment'] + \
               outflows['medical_insurance'] + outflows['pension_contribution'] + \
               outflows['misc'])

    # sposta il calcolo dei rendimenti degli investimenti al di sopra di quello
    # del reddito e usa la nuova variabile "assets" come base di investimento
    market_return = np.random.normal(variables['avg_monthly_market_returns'],
                                     variables['avg_monthly_market_volatility'],
                                     1)[0]

    investment_return = (assets * market_return) * (1 - variables['tax_on_investment_gains'])

    investment_gains_storage.append(investment_return)

    # aggiunge i rendimenti degli investimenti alla variabile "assets"
    assets += investment_return

    income = (inflows['active_annual_income'] * (1 - variables['tax_on_active_income_gains'])) / 12
    income_gains_storage.append(income)

    if (month % 12 == 0):
        inflows['active_annual_income'] *= (1 + (variables['avg_ann_income_raise']))

        # incrementa i flussi di uscita dell'inflazione annuale
        outflows['rent'] *= (1 + (variables['avg_ann_inflation']))
        outflows['credit_card_payment'] *= (1 + (variables['avg_ann_inflation']))
        outflows['medical_insurance'] *= (1 + (variables['avg_ann_inflation']))
        outflows['pension_contribution'] *= (1 + (variables['avg_ann_inflation']))
        outflows['misc'] *= (1 + (variables['avg_ann_inflation']))

    # aggiunge il guadagno del reddito alla variabile "assets"
    assets += income

    # calcolare il valore degli asset alla fine del periodo
    assets_ending = assets
    # memorizza il valore finale degli asset
    assets_ending_list.append(assets_ending)

plt.plot(pd.Series(assets_ending_list))
plt.xlabel('Month')
plt.ylabel('Ending Asset Value')
plt.show()

#############################################################################################################

start, end = "2000-12-31", "2020-01-01"
tickers = ["^SP500TR"]
sp = pd.DataFrame([yf.download(ticker, start, end).loc[:, 'Adj Close'] for ticker in tickers],
                  index=tickers).T.fillna(method='ffill')

sp_monthly_pct_return = sp.resample('M').last().pct_change().mean().values[0]
sp_monthly_std_dev = sp.resample('M').last().pct_change().std().values[0]

inflows = {'active_annual_income': 50_000,
           'starting_assets': 75_000}
# aggiungere il dizionario outflows
outflows = {'rent': 1500,
            'credit_card_payment': 750,
            'medical_insurance': 1250,
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
# crea una variabile per segnalare se siamo diventati finanziariamente "rovinati"
ruined = False

for month in range(months):

    if assets_ending_list:
        assets_starting_list.append(assets_ending_list[-1])

    assets = assets_starting_list[-1]
    assets -= (outflows['rent'] + outflows['credit_card_payment'] + \
               outflows['medical_insurance'] + outflows['pension_contribution'] + \
               outflows['misc'])

    # controlla se la base patrimoniale ha valore positivo. Se negativo
    # si imposta il flag "ruined" a 1 e si termina la simulazione
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