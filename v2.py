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
from neuralprophet import NeuralProphet

warnings.filterwarnings("ignore")


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


def avg_exp_hist_return(df, exp_returns_funcs):
    E = exp_returns_funcs[0](df)

    for i in range(1, len(exp_returns_funcs)):
        E = E.combine(exp_returns_funcs[i](df), lambda x, y: (x+y)/(i/i+1))
    return E


def avg_cov(df, cov_funcs):
    avg = sum([func(df).stack() for func in cov_funcs])/len(cov_funcs)
    S = avg.unstack()

    S = fix_nonpositive_semidefinite(S, fix_method='spectral')
    S = fix_nonpositive_semidefinite(S, fix_method='diag')
    return S


def mvo_opt(df, exp_returns_func, cov_func, objective, sector_mapper, sector_lower, sector_upper, constraint=None):
    mu = exp_returns_func(df)
    S = cov_func(df)

    ef = EfficientFrontier(mu, S)
    if sector_mapper is not None or sector_lower is not None or sector_upper is not None:
        ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    if constraint:
       ef.add_constraint(constraint)

    if objective == 'max_sharpe':
        ef.max_sharpe()
    elif objective == 'min_volatility':
        ef.min_volatility()

    weights = ef.clean_weights()
    return weights


def es_opt(df, exp_returns_func, objective, sector_mapper, sector_lower, sector_upper, constraint=None, target=0.10):
    mu = exp_returns_func(df)
    historical_returns = returns_from_prices(df)

    es = EfficientSemivariance(mu, historical_returns, verbose=False)
    es.add_objective(objective_functions.L2_reg, gamma=0.5)
    if sector_mapper is not None or sector_lower is not None or sector_upper is not None:
        es.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    if constraint:
        es.add_constraint(constraint)

    if objective == 'efficient_risk':
        es.efficient_risk(target)
    elif objective == 'min_semivariance':
        es.min_semivariance()
    elif objective == 'max_sharpe':
        es.max_sharpe()

    weights = es.clean_weights()
    return weights


def cvar_opt(df, exp_returns_func, objective, sector_mapper, sector_lower, sector_upper, constraint=None, target=0.10):
    mu = exp_returns_func(df)
    returns = returns_from_prices(df)

    ec = EfficientCVaR(mu, returns)
    ec.add_objective(objective_functions.L2_reg, gamma=1)
    if sector_mapper is not None or sector_lower is not None or sector_upper is not None:
        ec.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    if constraint:
        ec.add_constraint(constraint)

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


def bl_opt(df, viewdict, mcaps, cov_func, objective, sector_mapper, sector_lower, sector_upper, constraint=None):
    cov_matrix = cov_func(df)
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict, pi="market", market_caps=mcaps, omega="default")
    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()

    ef = EfficientFrontier(ret_bl, S_bl)
    if sector_mapper is not None or sector_lower is not None or sector_upper is not None:
        ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    if constraint:
        ef.add_constraint(constraint)
    if objective == 'max_sharpe':
        ef.max_sharpe()
    elif objective == 'min_volatility':
        ef.min_volatility()

    weights = ef.clean_weights()
    return weights


def avg_portfolios(allocations, weights=None):
    if weights:
        allocations = [weight_allocation(allocations[i], weights[i]) for i in range(len(allocations))]
    df = pd.DataFrame(allocations)
    avg = dict(df.mean())
    return avg


def weight_allocation(allocation, weight):
    return {k: v*weight for k, v in allocation.items()}


def mean_period_bl_opt(prices, viewdict, mcaps, cov_funcs, objective, sector_mapper=None, sector_lower=None, sector_upper=None, constraint=None):
    allocations = []

    for p in periods:
        df = get_period_df(prices, p)

        for cov_func in cov_funcs:
            try:
                opt_port = bl_opt(df, viewdict, mcaps, lambda df: avg_cov(df, cov_funcs), objective, sector_mapper, sector_lower, sector_upper, constraint)
            except:
                print(traceback.format_exc())
            try:
                opt_port = bl_opt(df, viewdict, mcaps, cov_func, objective, sector_mapper, sector_lower, sector_upper, constraint)
                allocations.append(opt_port)
            except:
                print(traceback.format_exc())

    mean = avg_portfolios(allocations)
    print("## MEAN BL OPT:")
    sort_print_dictionary(mean)
    run_benchmark_suite(prices, mean)
    return mean


def mean_period_mvo_opt(prices, exp_returns_funcs, cov_funcs, objective, sector_mapper=None, sector_lower=None, sector_upper=None, constraint=None):
    allocations = []

    for p in periods:
        df = get_period_df(prices, p)

        for er_func in exp_returns_funcs:
            try:
                opt_port = mvo_opt(df, er_func, lambda df: avg_cov(df, cov_funcs), objective, sector_mapper, sector_lower, sector_upper, constraint)
                allocations.append(opt_port)
            except Exception as e:
                print(traceback.format_exc())

            for cov_func in cov_funcs:
                try:
                    opt_port = mvo_opt(df, lambda df: avg_exp_hist_return(df, exp_returns_funcs), cov_func, objective, sector_mapper, sector_lower, sector_upper, constraint)
                    allocations.append(opt_port)
                except:
                    print(traceback.format_exc())
                try:
                    opt_port = mvo_opt(df, er_func, cov_func, objective, sector_mapper, sector_lower, sector_upper, constraint)
                    allocations.append(opt_port)
                except:
                    print(traceback.format_exc())

    mean = avg_portfolios(allocations)
    print("## MEAN MVO OPT:")
    sort_print_dictionary(mean)
    run_benchmark_suite(prices, mean)
    return mean


def mean_period_es_opt(prices, exp_returns_funcs, objective, sector_mapper, sector_lower, sector_upper, constraint=None, target=None):
    allocations = []

    for p in periods:
        df = get_period_df(prices, p)

        try:
            opt_port = es_opt(df, lambda df: avg_exp_hist_return(df, exp_returns_funcs), objective, sector_mapper, sector_lower, sector_upper, constraint, target)
            allocations.append(opt_port)
        except:
            print(traceback.format_exc())

        for er_func in exp_returns_funcs:
            try:
                opt_port = es_opt(df, er_func, objective, sector_mapper, sector_lower, sector_upper, constraint, target)
                allocations.append(opt_port)
            except:
                print(traceback.format_exc())

    mean = avg_portfolios(allocations)
    print("## MEAN ES OPT:")
    sort_print_dictionary(mean)
    run_benchmark_suite(prices, mean)
    return mean


def mean_period_cvar_opt(prices, exp_returns_funcs, objective, sector_mapper, sector_lower, sector_upper, constraint=None, target=None):
    allocations = []

    for p in periods:
        df = get_period_df(prices, p)

        try:
            opt_port = cvar_opt(df, lambda df: avg_exp_hist_return(df, exp_returns_funcs), objective, sector_mapper, sector_lower, sector_upper, constraint, target)
            allocations.append(opt_port)
        except:
            print(traceback.format_exc())

        for er_func in exp_returns_funcs:
            try:
                opt_port = cvar_opt(df, er_func, objective, sector_mapper, sector_lower, sector_upper, constraint, target)
                allocations.append(opt_port)
            except:
                print(traceback.format_exc())

    mean = avg_portfolios(allocations)
    print("## MEAN CVAR OPT:")
    sort_print_dictionary(mean)
    run_benchmark_suite(prices, mean)
    return mean


def mean_period_hrp_opt(prices):
    allocations = []

    for p in periods:
        df = get_period_df(prices, p)
        try:
            opt_port = hrp_opt(df)
            allocations.append(opt_port)
        except:
            print(traceback.format_exc())

    mean = avg_portfolios(allocations)
    print("## MEAN HRP OPT:")
    sort_print_dictionary(mean)
    run_benchmark_suite(prices, mean)
    return mean


def select_assets(prices, exp_returns_funcs, cov_funcs, viewdict, mcaps, num_assets, sector_mapper, sector_lower, sector_upper):
    print('### ASSET SELECTION')
    allocations = {
        "hrp": mean_period_hrp_opt(prices),
        "es": mean_period_es_opt(prices, exp_returns_funcs, 'efficient_risk', sector_mapper, sector_lower, sector_upper, None, 0.10),
        "cvar": mean_period_cvar_opt(prices, exp_returns_funcs, 'efficient_risk', sector_mapper, sector_lower, sector_upper, None, 0.10),
        "mvo": mean_period_mvo_opt(prices, exp_returns_funcs, cov_funcs, 'max_sharpe', None, None, None),
    }
    if len(viewdict) > 0:
        allocations["bl"] = mean_period_bl_opt(prices, viewdict, mcaps, cov_funcs, 'max_sharpe', sector_mapper, sector_lower, sector_upper)
        #allocations["bl_min"] = mean_period_bl_opt(prices, viewdict, mcaps, cov_funcs, 'min_volatility', None, None, None)


    mean = avg_portfolios(list(allocations.values()))#, [1, 0,5, 0.5, 1, 1])
    allocations["mean"] = mean
    print("## MEAN ASSET SELECT OPT:")
    sort_print_dictionary(mean)
    run_benchmark_suite(prices, mean)
    plot_ef_allocations(allocations, prices, "Asset selection")

    result = []
    for i in range(min(num_assets, len(prices))):
        highest_valued_key = max(mean, key=mean.get)
        result.append(highest_valued_key)
        mean.pop(highest_valued_key)

    return result


def optimize_portfolio(prices, exp_returns_funcs, cov_funcs, viewdict, mcaps, sector_mapper, sector_lower, sector_upper):
    print("### OPTIMAL PORTFOLIO:")
    allocations = {
        "hrp": mean_period_hrp_opt(prices),
        "es": mean_period_es_opt(prices, exp_returns_funcs, 'efficient_risk', None, None, None, lambda x : x >= 0.05, 0.10),
        "cvar": mean_period_cvar_opt(prices, exp_returns_funcs, 'efficient_risk', sector_mapper, sector_lower, sector_upper, lambda x: x >= 0.05, 0.10),
        "mvo": mean_period_mvo_opt(prices, exp_returns_funcs, cov_funcs, 'max_sharpe', None, None, None, lambda x : x >= 0.05),
    }
    if len(viewdict) > 0:
        allocations["bl"] = mean_period_bl_opt(prices, viewdict, mcaps, cov_funcs, 'max_sharpe', None, None, None, lambda x: x >= 0.05)
        #allocations["bl_min"] = mean_period_bl_opt(prices, viewdict, mcaps, cov_funcs, 'min_volatility', None, None, None, lambda x: x >= 0.05)


    mean = avg_portfolios(list(allocations.values()))#, [1, 0,5, 0.5, 1, 1])
    allocations["mean"] = mean
    print("## MEAN OPTIMAL PORTFOLIO")
    sort_print_dictionary(mean)
    run_benchmark_suite(prices, mean)
    plot_ef_allocations(allocations, prices, "Optimal portfolio")
    return mean



def filter_dict(d, keys):
    ret = dict(filter(lambda elem: elem[0] in set(keys), d.items()))
    return ret


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


def sort_print_dictionary(dictionary):
    sorted_dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    df = pd.DataFrame(sorted_dictionary, index=["weight"])

    for i in range(0, len(df.columns), 10):
        data = df.iloc[:,i:i+10]
        print(data)
    print()


def plot_ef_allocations(allocations, prices, title="Efficient Frontier with portfolio and assets"):
    
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    

    for i, p in enumerate(["1mo", "3mo", "6mo", "1y"]):
        df = get_period_df(prices, p)
        mu = ema_historical_return(df)
        S = CovarianceShrinkage(df).ledoit_wolf()

        ef = EfficientFrontier(mu, S)
        ef.max_sharpe()
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
        
        ax.scatter(ms_std, ms_ret, marker="*", s=100, c="g", label="max sharpe")
    
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


def main(periods, num_stocks, viewdict, sector_mapper, sector_lower, sector_upper):
    print("BL viewdict:")
    sort_print_dictionary(viewdict)
    
    prices, mcaps = get_data(sector_mapper.keys(), periods)

    exp_returns_funcs = [mean_historical_return, ema_historical_return, capm_return]
    cov_funcs = [
        lambda df: CovarianceShrinkage(df).ledoit_wolf(shrinkage_target="single_factor"),
        lambda df: CovarianceShrinkage(df).ledoit_wolf(shrinkage_target="constant_correlation"),
        lambda df: CovarianceShrinkage(df).ledoit_wolf(shrinkage_target="constant_variance"),
        lambda df: CovarianceShrinkage(df).oracle_approximating(),
        exp_cov,
        sample_cov
    ]

    stock_picks = select_assets(prices, exp_returns_funcs, cov_funcs, viewdict, mcaps, num_stocks, sector_mapper, sector_lower, sector_upper)
    prices_filtered = filter_dict(prices, stock_picks)
    viewdict_filtered = filter_dict(viewdict, stock_picks)
    mcaps_filtered = filter_dict(mcaps, stock_picks)

    sector_mapper = filter_dict(sector_mapper, stock_picks)

    opt_port = optimize_portfolio(prices_filtered, exp_returns_funcs, cov_funcs, viewdict_filtered, mcaps_filtered, sector_mapper, sector_lower, sector_upper)
    print_sector_summary(opt_port, sector_mapper)
    print(opt_port)
    print()

    

    



if __name__ == '__main__':
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

    sector_lower = {
        'it': 0.05,
        'energi': 0.05,
        'finans': 0.05,
        'helsevern': 0.05,
        'materialer': 0.05
    }

    sector_upper = {
        'industri': 0.2,
        'konsumvarer': 0.2,
        'finans': 0.2,
        'energi': 0.2
    }


    periods = ["1mo", "2mo", "3mo", "4mo", "5mo", "6mo", "7mo", "8mo", "9mo", "10mo", "11mo", "1y"]

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
        "NHY.OL": get_view("NHY.OL", (73+70)/2),
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
        "ATEA.OL": get_view("ATEA.OL", 190)
    }



    num_stocks = 15

    main(periods, num_stocks, viewdict, sector_mapper, sector_lower, sector_upper)

