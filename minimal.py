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

periods = ["1mo", "3mo", "6mo", "1y", "3y"]
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
    'SCHA.OL': 'it',
    'MPCC.OL': 'industri',
    'WSTEP.OL': 'it',
    'NSKOG.OL': 'materialer'
}




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


def get_view(stock, goal):
    r = requests.get(f'http://127.0.0.1:5000/last_price?stock={stock}')
    price = float(r.text)
    return round((goal - price)/price, 3)


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


def sort_print_dictionary(dictionary):
    sorted_dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    df = pd.DataFrame(sorted_dictionary, index=["weight"])

    for i in range(0, len(df.columns), 10):
        data = df.iloc[:,i:i+10]
        print(data)
    print()



def hrp_opt(df):
    rets = returns_from_prices(df)
    covtrix = exp_cov(df)
    hrp = HRPOpt(returns=rets, cov_matrix=covtrix)
    hrp.optimize()
    weights = hrp.clean_weights()
    return weights


def bl_ret_S(df, cov_func, viewdict, mcaps):
    cov_matrix = cov_func(df)
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict, pi="market", market_caps=mcaps, omega="default")
    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()
    return ret_bl, S_bl


def plot_ef_allocations(allocations, prices, ret_func, cov_func, title="Efficient Frontier with portfolio and assets", bl=False, viewdict=None, mcaps=None):
    
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    

    for i, p in enumerate(["3mo", "6mo", "1y", "3y"]):

        df = get_period_df(prices, p)
        
        if bl:
            mu, S = bl_ret_S(df, cov_func, viewdict, mcaps)
        else:
            mu = ret_func(df)
            S = cov_func(df)
        
        ef = EfficientFrontier(mu, S)

        ef.min_volatility()
        ms_ret, ms_std, _ = ef.portfolio_performance(verbose=False)

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

        ax.scatter(ms_std, ms_ret, marker="*", s=100, c="g", label="min_volatility")
        
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1,1))
    
    #plt.tight_layout()
    plt.savefig(f"{title}.png")


def amend_allocation(allocation, all_stocks):
    ret = {}
    for stock in all_stocks:
        ret[stock] = allocation[stock] if stock in allocation.keys() else 0

    return ret


def print_sector_summary(allocation, sector_mapper):
    for sector in set(sector_mapper.values()):
        total_weight = 0
        for t, w in allocation.items():
            if sector_mapper[t] == sector:
                total_weight += w
        print(f"{sector}: {total_weight:.3f}")
    print()



def main():
	stocks = ['NSKOG.OL', 'ULTI.OL', 'OLT.OL', 'MEDI.OL', 'SRBNK.OL', 'LSG.OL', 'SCHA.OL', 'FRO.OL', 'KOG.OL', 'AKSO.OL', 'ATEA.OL', 'ADE.OL']

	viewdict = {
	    "ACC.OL": get_view("ACC.OL", 33),
	    "AKER.OL": get_view("AKER.OL", 870),
	    "AKH.OL": get_view("AKH.OL", 35),
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
	    "NHY.OL": get_view("NHY.OL", (73+70+82)/3),
	    "NOD.OL": get_view("NOD.OL", 350),
	    "SALM.OL": get_view("SALM.OL", 650),
	    "SALME.OL": get_view("SALME.OL", 10),
	    "SCATC.OL": get_view("SCATC.OL", 210),
	    "YAR.OL": get_view("YAR.OL", 400),
	    "SCHA.OL": get_view("SCHA.OL", 504),
	    "ADE.OL": get_view("ADE.OL", 185),
	    "MPCC.OL": get_view("MPCC.OL", 27),
	    "ORK.OL": get_view("ORK.OL", 90),
	    "WSTEP.OL": get_view("WSTEP.OL", 40),
	    "NSKOG.OL": get_view("NSKOG.OL", 60),
	    "ATEA.OL": get_view("ATEA.OL", 190),
	    "KOG.OL": get_view("KOG.OL", 315)
	}

	prices, _ = get_data(stocks, periods)
	df = get_period_df(prices, '3y')

	opt_port = hrp_opt(df)
	allocations = {
		'hrp': opt_port
	}

	all_stocks = list(sector_mapper.keys())
	all_prices, mcaps = get_data(all_stocks, periods)

	sort_print_dictionary(opt_port)
	plot_ef_allocations(allocations, all_prices, ema_historical_return, exp_cov, "minimal HRP min mvo")
	plot_ef_allocations(allocations, all_prices, ema_historical_return, exp_cov, "minimal HRP bl", bl=True, viewdict=viewdict, mcaps=mcaps)
	print_sector_summary(opt_port, sector_mapper)
	run_benchmark_suite(all_prices, amend_allocation(opt_port, all_stocks))
	print('viewdict')
	sort_print_dictionary(viewdict)

if __name__ == '__main__':
	main()