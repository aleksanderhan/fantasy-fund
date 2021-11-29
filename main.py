import yfinance as yf
import pandas as pd
import numpy as np
from functools import reduce
from pypfopt.expected_returns import ema_historical_return, capm_return
from pypfopt.risk_models import CovarianceShrinkage, exp_cov, sample_cov, fix_nonpositive_semidefinite
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.black_litterman import BlackLittermanModel
from neuralprophet import NeuralProphet


all_tickets = ["CRAYN.OL", "OLT.OL", "NOD.OL", "AKER.OL", "BWO.OL", "EQNR.OL", "SRBNK.OL", "KOG.OL", "NHY.OL", "FRO.OL", "ACC.OL", "MEDI.OL", "AFG.OL", "RECSI.OL", "DNO.OL", "ENTRA.OL", "KAHOT.OL", "LSG.OL", "MOWI.OL", "SALM.OL", "DNB.OL", "MULTI.OL", "PARB.OL", "YAR.OL", "ASTK.OL", "SCATC.OL", "AKH.OL", "NAPA.OL", "ORK.OL", "TEL.OL", "AKSO.OL", "BEWI.OL", "BELCO.OL", "KID.OL", "SALME.OL", "NORBT.OL"]
periods = ["1wk", "2wk", "3wk", "1mo", "2mo", "3mo", "4mo", "5mo", "6mo", "7mo", "8mo", "9mo", "10mo", "11mo", "1y"]

#periods = ["1y", "1wk", "2wk", "3wk", "1mo"]



def get_history(hist, ticket, average=False):	
	hist.drop('Volume', axis=1, inplace=True)
	hist.drop('Dividends', axis=1, inplace=True)
	hist.drop('Stock Splits', axis=1, inplace=True)
	if not average:
		hist.drop('Open', axis=1, inplace=True)
		hist.drop('High', axis=1, inplace=True)
		hist.drop('Low', axis=1, inplace=True)

	av_row = hist.mean(axis=1)
	return av_row.to_frame(ticket)


def get_df(history):
	df = reduce(lambda left, right: pd.merge_ordered(left, right, on='Date'), history.values())
	df = df.set_index('Date')
	return df


def get_data(tickets, periods):
	data = {p: {} for p in periods}
	mcaps = {}

	for i, ticket in enumerate(tickets):
		print('Downloading data for', ticket, i+1, 'of', len(tickets))
		ticker = yf.Ticker(ticket)
		mcaps[ticket] = ticker.info["marketCap"]

		for p in periods:
			hist = ticker.history(period=p)
			data[p][ticket] = get_history(hist, ticket)

	return data, mcaps


def get_viewdict(tickets, data):
	viewdict = {}

	for ticket in tickets:
		view = get_view(ticket, data)
		viewdict[ticket] = view

	return viewdict


def get_view(ticket, data):
	print('Creating view for', ticket)
	m = NeuralProphet(epochs=10, num_hidden_layers=2, n_forecasts=7)
	
	df = data["1y"][ticket]
	temp = df.reset_index(level=0)
	temp.rename(columns={"Date": "ds"}, inplace=True)
	temp.rename(columns={ticket: "y"}, inplace=True)
	
	now = temp.tail(1)
	now.reset_index(level=0, inplace=True)
	y0 = now['y'].loc[0]

	metrics = m.fit(temp, freq="D")
	future = m.make_future_dataframe(temp, periods=7)
	forecast = m.predict(future)
	forecast.head()

	prediction = forecast.tail(1)
	prediction.reset_index(level=0, inplace=True)
	yhat = prediction['yhat1'].loc[0]

	return round((yhat-y0)/y0, 3)
	
	


def get_expected_returns(df, tickets, viewdict, mcaps):
	emhr = ema_historical_return(df)
	capmr = capm_return(df)

	E = emhr.combine(capmr, lambda x, y: (x+y)/2)

	S = get_cov(df)
	bl = BlackLittermanModel(S, absolute_views=viewdict, pi="market", market_caps=mcaps, omega="default")
	rets = bl.bl_returns()
	print('RETS', rets)
	E = E.combine(rets, lambda x, y: (x+y))*(2/3)

	return E


def get_cov(df):
	lw = CovarianceShrinkage(df).ledoit_wolf()
	oa = CovarianceShrinkage(df).oracle_approximating()
	ec = exp_cov(df)

	S = (lw.stack()+oa.stack()+ec.stack())/3
	S = S.unstack()

	S = fix_nonpositive_semidefinite(S, fix_method='spectral')
	S = fix_nonpositive_semidefinite(S, fix_method='diag')
	return S
	


def opt_port(df, add_constraint, tickets, viewdict, mcaps, target):
	mu = get_expected_returns(df, tickets, viewdict, mcaps)
	S = get_cov(df)

	ef = EfficientFrontier(mu, S)
	if add_constraint:
		ef.add_constraint(lambda x : x >= 0.05)

	if target == 'max_sharpe':
		ef.max_sharpe()
	elif target == 'min_volatility':
		ef.min_volatility()

	cleaned_weights = ef.clean_weights()
	
	print()
	print(cleaned_weights)
	ef.portfolio_performance(verbose=True)
	print()

	return cleaned_weights



def stacked_average_optimization(data, tickets, add_constraint, viewdict, mcaps, target):
	optimized_portfolio = {t: 0 for t in tickets}

	for p in periods:
		print("period", p)
		df = get_df(data[p])
		port = opt_port(df, add_constraint, tickets, viewdict, mcaps, target)

		for k, v in port.items():
			optimized_portfolio[k] += v

	return {t: optimized_portfolio[t]/len(periods) for t in tickets}



def get_standard_benchmark(df):
	mu = ema_historical_return(df)
	S = CovarianceShrinkage(df).ledoit_wolf()
	ef = EfficientFrontier(mu, S)
	ef.min_volatility()
	cleaned_weights = ef.clean_weights()

	print()
	print(cleaned_weights)
	ef.portfolio_performance(verbose=True)
	print()

	return cleaned_weights



def main():

	data, mcaps = get_data(all_tickets, periods)
	viewdict = get_viewdict(all_tickets, data)

	all_asset_opt_port = stacked_average_optimization(data, all_tickets, False, viewdict, mcaps, 'max_sharpe')

	tickets = []
	for i in range(min(len(all_asset_opt_port), 15)):
		highest_valued_key = max(all_asset_opt_port, key=all_asset_opt_port.get)
		tickets.append(highest_valued_key)
		all_asset_opt_port.pop(highest_valued_key)

	filtered_data = {}
	for p in periods:
		item = {}
		for k, v in data[p].items():
			if k in set(tickets):
				item[k] = v
		filtered_data[p] = item

	filtered_viewdict = dict(filter(lambda elem: elem[0] in set(tickets), viewdict.items()))
	filtered_mcaps = dict(filter(lambda elem: elem[0] in set(tickets), mcaps.items()))

	optimized_portfolio = stacked_average_optimization(filtered_data, tickets, True, filtered_viewdict, filtered_mcaps, 'min_volatility')

	for p in periods:
		print("period", p)
		df = get_df(filtered_data[p])
		mu = get_expected_returns(df, tickets, filtered_viewdict, filtered_mcaps)
		S = get_cov(df)
		ef = EfficientFrontier(mu, S)
		ef.set_weights(optimized_portfolio)
		ef.portfolio_performance(verbose=True)
		print()
	
	for k, v in optimized_portfolio.items():
		print(k, round(v, 2))

	print('viewdict', viewdict)
	print()
	
	for p in periods:
		print('period', p)
		df = get_df(data[p])
		get_standard_benchmark(df)



if __name__ == '__main__':
	main()
