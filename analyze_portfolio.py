import traceback
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from functools import reduce
from pypfopt.expected_returns import mean_historical_return, ema_historical_return, capm_return, returns_from_prices
from pypfopt.risk_models import CovarianceShrinkage, exp_cov, sample_cov, fix_nonpositive_semidefinite
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.black_litterman import BlackLittermanModel, market_implied_risk_aversion
from pypfopt import EfficientSemivariance, HRPOpt, EfficientCVaR, CLA, plotting, objective_functions

def get_view(stock, goal):
    r = requests.get(f'http://127.0.0.1:5000/last_price?stock={stock}')
    price = float(r.text)
    return round((goal - price)/price, 3)


periods = ["1mo", "2mo", "3mo", "4mo", "5mo", "6mo", "7mo", "8mo", "9mo", "10mo", "11mo", "1y"]

viewdict = {
    "ACC.OL": get_view("ACC.OL", 33),
    "AKER.OL": get_view("AKER.OL", 870),
    "AKH.OL": get_view("AKH.OL", 25),
    "AKSO.OL": get_view("AKSO.OL", 25),
    "BELCO.OL": get_view("BELCO.OL", 19),
    "BEWI.OL": get_view("BEWI.OL", 65),
    "CRAYN.OL": get_view("CRAYN.OL", (190+178)/2),
    "DNB.OL": get_view("DNB.OL", 213),
    "EQNR.OL": get_view("EQNR.OL", 260),
    "FRO.OL": get_view("FRO.OL", 105),
    "LSG.OL": get_view("LSG.OL", 105),
    "MOWI.OL": get_view("MOWI.OL", 270),
    "MULTI.OL": get_view("MULTI.OL", 180),
    "NHY.OL": get_view("NHY.OL", (73+70)/2),
    "NOD.OL": get_view("NOD.OL", 350),
    "SALM.OL": get_view("SALM.OL", 650),
    "SALME.OL": get_view("SALME.OL", 10),
    "SCATC.OL": get_view("SCATC.OL", 210),
    "YAR.OL": get_view("YAR.OL", 400),
    "SCHA.OL": get_view("SCHA.OL", 504),
    "ADE.OL": get_view("ADE.OL", 185)
}

sector_mapper = {
    'ACC.OL': 'energi', 
    'AFG.OL': 'industri',
    'AKER.OL': 'finans',
    'AKH.OL': 'forsyning',
    'AKRBP.OL': 'energi',
    'AKSO.OL': 'energi',
    'ASTK.OL': 'it',
    'ATEA.OL': 'it',
    'AZT.OL': 'helsevern',
    'BELCO.OL': 'industri',
    'BEWI.OL': 'industri',
    'BWO.OL': 'energi',
    'CRAYN.OL': 'it',
    'DNB.OL': 'finans',
    'DNO.OL': 'energi',
    'ENTRA.OL': 'eiendom',
    'EQNR.OL': 'energi',
    'FRO.OL': 'industri',
    'KAHOT.OL': 'forbruksvarer',
    'KID.OL': 'forbruksvarer',
    'KOG.OL': 'industri',
    'LSG.OL': 'konsumvarer',
    'MEDI.OL': 'helsevern',
    'MOWI.OL': 'konsumvarer',
    'MULTI.OL': 'industri',
    'NAPA.OL': 'telekom',
    'NHY.OL': 'materialer',
    'NOD.OL': 'it',
    'NORBT.OL': 'industri',
    'OET.OL': 'industri',
    'OLT.OL': 'eiendom',
    'ORK.OL': 'konsumvarer',
    'PARB.OL': 'finans',
    'PEXIP.OL': 'it',
    'RECSI.OL': 'materialer',
    'SALM.OL': 'konsumvarer',
    'SALME.OL': 'konsumvarer',
    'SCATC.OL': 'energi',
    'SRBNK.OL': 'finans',
    'TEL.OL': 'telekom',
    'ULTI.OL': 'helsevern',
    'YAR.OL': 'materialer',
    'ADE.OL': 'forbruksvarer',
    'GJF.OL': 'finans',
    'SCHA.OL': 'it'
}

exp_returns_funcs = [mean_historical_return, ema_historical_return, capm_return]
cov_funcs = {
    'single_factor': lambda df: CovarianceShrinkage(df).ledoit_wolf(shrinkage_target="single_factor"),
    'constant_correlation': lambda df: CovarianceShrinkage(df).ledoit_wolf(shrinkage_target="constant_correlation"),
    'constant_variance': lambda df: CovarianceShrinkage(df).ledoit_wolf(shrinkage_target="constant_variance"),
    'oracle_approximating': lambda df: CovarianceShrinkage(df).oracle_approximating(),
}


def get_data(stocks, periods):
    prices = {s: {} for s in stocks}
    mcaps = {}

    for i, stock in enumerate(stocks):
        print("Downloading data for", stock, i+1, "of", len(stocks))
        r_mcap = requests.get(f'http://127.0.0.1:5000/mcap?stock={stock}')
        mcaps[stock] = int(r_mcap.text)

        for p in periods:
            r_prices = requests.get(f'http://127.0.0.1:5000/prices?stock={stock}&period={p}')
            prices[stock][p] = pd.DataFrame.from_dict(r_prices.json())

    print()
    return prices, mcaps


def get_period_df(prices, period):
    for stock in prices.keys():
        for p in periods:
            prices[stock][p]['Date'] = prices[stock][p].index

    values = [prices[stock][period] for stock in prices.keys()]
    df = reduce(lambda left, right: pd.merge_ordered(left, right, on='Date'), values)
    df = df.set_index('Date')
    return df


def plot_ef_allocations(allocations, prices, ret_func, cov_func, title="Efficient Frontier with portfolio and assets", bl=False, viewdict=None, mcaps=None):
    
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    

    for i, p in enumerate(["1mo", "3mo", "6mo", "1y"]):

        df = get_period_df(prices, p)
        
        if bl:
            mu, S = bl_ret_S(df, cov_func, viewdict, mcaps)
        else:
            mu = ret_func(df)
            S = cov_func(df)
        
        ef = EfficientFrontier(mu, S)
        #ef.max_sharpe()
        #ms_ret, ms_std, _ = ef.portfolio_performance(verbose=False)

        alloc_rets, alloc_stds = {}, {}
        for key, alloc in allocations.items():
            alloc = amend_allocation(alloc, df.columns)
            ef.set_weights(alloc)
            ret, std, _ = ef.portfolio_performance(verbose=False)
            alloc_rets[key] = ret
            alloc_stds[key] = std

        x, y = ((0, 0) if i%3 == 0 else (1, 1)) if i%2 == 0 else ((0, 1) if i%3 == 0 else (1, 0))
        ax = axs[x, y]
        ax.set_title(f"period {p}")

        plotting.plot_efficient_frontier(CLA(mu, S), ax=ax, show_assets=True)

        for key in allocations.keys():
            ax.scatter(alloc_stds[key], alloc_rets[key], marker="*", s=100, label=key)
        
        #ax.scatter(ms_std, ms_ret, marker="*", s=100, c="g", label="max sharpe")
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1,1))
    
    #plt.tight_layout()
    plt.savefig(f"{title}.png")


def plot_cvar(allocations, prices, ret_func, cov_func, viewdict, mcaps, period):
    
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"period {period}")

    for i, p in enumerate(["1mo", "3mo", "6mo", "1y"]):

        df = get_period_df(prices, p)

        mu = ret_func(df)
        S = cov_func(df)
        
        ef = EfficientFrontier(mu, S)

        alloc_rets, alloc_stds = {}, {}
        for key, alloc in allocations.items():
            alloc = amend_allocation(alloc, df.columns)
            ef.set_weights(alloc)
            ret, std, _ = ef.portfolio_performance(verbose=False)
            alloc_rets[key] = ret
            alloc_stds[key] = std

        x, y = ((0, 0) if i%3 == 0 else (1, 1)) if i%2 == 0 else ((0, 1) if i%3 == 0 else (1, 0))
        ax = axs[x, y]
        ax.set_title()

        plotting.plot_efficient_frontier(CLA(mu, S), ax=ax, show_assets=True)

        for key in allocations.keys():
            ax.scatter(alloc_stds[key], alloc_rets[key], marker="*", s=100, label=key)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1,1))
    
    #plt.tight_layout()
    plt.savefig(f"{title}.png")


def run_benchmark_suite(prices, allocation):
    performance = []
    for p in periods:
        df = get_period_df(prices, p)
        perf = benchmark_allocation(df, allocation)
        performance.append(perf)

    columns = ["Expected annual return", "Annual volatility", "Sharpe ratio"]
    df = pd.DataFrame(performance, columns=columns, index=periods)
    print(df)
    mean = dict(df.mean(axis=0))
    std = dict(df.std(axis=0))
    median = dict(df.median(axis=0))

    ret = pd.DataFrame([[mean[column], std[column], median[column]] for column in columns], columns=["mean", "std", "median"], index=columns)
    print(ret)
    print()
    return ret


def benchmark_allocation(df, allocation, verbose=False):
    mu = ema_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    ef.set_weights(allocation)

    perf = ef.portfolio_performance(verbose=verbose)
    return perf


def amend_allocation(allocation, all_stocks):
    ret = {}
    for stock in all_stocks:
        ret[stock] = allocation[stock] if stock in allocation.keys() else 0

    return ret

def mvo_opt(df, exp_returns_func, cov_func, objective):
    mu = exp_returns_func(df)
    S = cov_func(df)

    ef = EfficientFrontier(mu, S)

    if objective == 'max_sharpe':
        ef.max_sharpe()
    elif objective == 'min_volatility':
        ef.min_volatility()

    weights = ef.clean_weights()
    return weights


def es_opt(df, exp_returns_func, objective, target=0.10):
    mu = exp_returns_func(df)
    historical_returns = returns_from_prices(df)

    es = EfficientSemivariance(mu, historical_returns, verbose=False)

    if objective == 'efficient_risk':
        es.efficient_risk(target)
    elif objective == 'min_semivariance':
        es.min_semivariance()

    weights = es.clean_weights()
    return weights


def cvar_opt(df, exp_returns_func, objective, target=0.10):
    mu = exp_returns_func(df)
    returns = returns_from_prices(df)

    ec = EfficientCVaR(mu, returns)

    if objective == 'efficient_risk':
        ec.efficient_risk(target)
    elif objective == 'min_cvar':
        ec.min_cvar()

    weights = ec.clean_weights()
    return weights


def hrp_opt(df):
    rets = returns_from_prices(df)
    hrp = HRPOpt(rets)
    hrp.optimize()
    weights = hrp.clean_weights()
    return weights


def bl_ret_S(df, cov_func, viewdict, mcaps):
    cov_matrix = cov_func(df)
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict, pi="market", market_caps=mcaps, omega="default")
    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()
    return ret_bl, S_bl


def bl_opt(df, viewdict, mcaps, cov_func, objective):
    ret_bl, S_bl = bl_ret_S(df, cov_func, viewdict, mcaps)

    cla = CLA(ret_bl, S_bl)
    if objective == 'max_sharpe':
        cla.max_sharpe()
    elif objective == 'min_volatility':
        cla.min_volatility()

    weights = cla.clean_weights()
    return weights


def sort_print_dictionary(dictionary):
    sorted_dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    df = pd.DataFrame(sorted_dictionary, index=["weight"])

    for i in range(0, len(df.columns), 10):
        data = df.iloc[:,i:i+10]
        print(data)
    print()


def main(prices, mcaps, Y):

    df = get_period_df(prices, "1y")

    allocations = {
        'mvo': mvo_opt(df, ema_historical_return, cov_funcs['constant_correlation'], 'max_sharpe'),
        #'es': es_opt(df, ema_historical_return, 'min_semivariance', target=0.10),
        #'cvar': cvar_opt(df, ema_historical_return, 'min_cvar', target=0.10),
        'hrp': hrp_opt(df),
        'bl': bl_opt(df, viewdict, mcaps, cov_funcs['oracle_approximating'], 'max_sharpe')
    }

    for i, y in enumerate(Y):
        allocations[f"y{i+1}"] = amend_allocation(y, sector_mapper.keys())

        sort_print_dictionary(y)
        print('sum', sum(y.values()))
    #run_benchmark_suite(prices, allocations['y'])

    
    plot_ef_allocations(allocations, prices, ema_historical_return, cov_funcs['single_factor'], "ema-single_factor-y1")

    plot_ef_allocations(allocations, prices, capm_return, cov_funcs['oracle_approximating'], "capm-oracle_approximating-y1")

    plot_ef_allocations(allocations, prices, ema_historical_return, cov_funcs['oracle_approximating'], "bl-ema-oracle-y1", True, viewdict, mcaps)

    plot_ef_allocations(allocations, prices, capm_return, cov_funcs['constant_correlation'], "bl-capm-const_corr-y1", True, viewdict, mcaps)



if __name__ == '__main__':
    prices, mcaps = get_data(sector_mapper.keys(), periods)

    y1 = {
        "LSG.OL": 0.16,
        "DNB.OL": 0.05,
        "TEL.OL": 0.06,
        "BELCO.OL": 0.07,
        "ORK.OL": 0.07,
        "NOD.OL": 0.05,
        "BEWI.OL": 0.12,
        "EQNR.OL": 0.07,
        "MEDI.OL": 0.05,
        "ULTI.OL": 0.05,
        "OLT.OL": 0.05,
        "SCATC": 0.05,
        "SRBNK.OL": 0.05,
        "FRO.OL": 0.05,
        "PARB.OL": 0.05
    }

    y2 = {
        "LSG.OL": 0.09,
        "BEWI.OL": 0.14,
        "ULTI.OL": 0.08,
        "ORK.OL": 0.07,
        "EQNR.OL": 0.05,
        "TEL.OL": 0.06,
        "NOD.OL": 0.05,
        "OLT.OL": 0.06,
        "BELCO.OL": 0.05,
        "PARB.OL": 0.06,
        "SRBNK.OL": 0.05,
        "ACC.OL": 0.09,
        "MEDI.OL": 0.06,
        "DNO.OL": 0.06,
        "DNB.OL": 0.07

    }

    y3 = {
        'ACC.OL': 0.05, 
        'BELCO.OL': 0.05, 
        'BEWI.OL': 0.14, 
        'DNO.OL': 0.05, 
        'EQNR.OL': 0.05, 
        'LSG.OL': 0.08, 
        'MEDI.OL': 0.06, 
        'NOD.OL': 0.05, 
        'OLT.OL': 0.07, 
        'ORK.OL': 0.08, 
        'PARB.OL': 0.07, 
        'SALME.OL': 0.06, 
        'SRBNK.OL': 0.05, 
        'ULTI.OL': 0.08, 
        'GJF.OL': 0.06,
    }

    y4 = {'ACC.OL': 0.05350725998807397, 'AKSO.OL': 0.051615670940170964, 'BELCO.OL': 0.0458299770174916, 'BEWI.OL': 0.11152343686642804, 'LSG.OL': 0.06393614450407481, 'MEDI.OL': 0.051136347967600894, 'NOD.OL': 0.051813643684158285, 'ORK.OL': 0.09759016830649977, 'PARB.OL': 0.06713213511230379, 'SALME.OL': 0.05215632655038764, 'SRBNK.OL': 0.057871861111111166, 'ULTI.OL': 0.0863380184605446, 'GJF.OL': 0.07403729521963831, 'MPCC.OL': 0.053372934704830066, 'NSKOG.OL': 0.08213674970184862}
    y5 = {'ACC.OL': 0.053496263034188064, 'AKSO.OL': 0.05161233482905986, 'BELCO.OL': 0.04582847863247867, 'BEWI.OL': 0.11104769145299136, 'LSG.OL': 0.06392540064102573, 'MEDI.OL': 0.05100648440170943, 'NOD.OL': 0.051761881410256484, 'ORK.OL': 0.09844056346153852, 'PARB.OL': 0.06730231217948723, 'SALME.OL': 0.052101011111111165, 'SRBNK.OL': 0.05785812500000005, 'ULTI.OL': 0.08603889914529912, 'GJF.OL': 0.07409654722222228, 'MPCC.OL': 0.05335175363247865, 'NSKOG.OL': 0.0821305440170941}


    Y = [y1, y2, y3, y4, y5]

    for y in Y:
        run_benchmark_suite(prices, amend_allocation(y, sector_mapper.keys()))

    main(prices, mcaps, Y)
