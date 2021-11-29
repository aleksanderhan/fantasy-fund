import yfinance as yf
from flask import Flask, request
from functools import lru_cache

app = Flask(__name__)


@lru_cache(maxsize=10000)
def get_history(stock, period, average=False):
	print('Downloading prices for', stock, "period", period)
	ticker = yf.Ticker(stock)
	hist = ticker.history(period=period)

	hist.drop('Volume', axis=1, inplace=True)
	hist.drop('Dividends', axis=1, inplace=True)
	hist.drop('Stock Splits', axis=1, inplace=True)
	if not average:
		hist.drop('Open', axis=1, inplace=True)
		hist.drop('High', axis=1, inplace=True)
		hist.drop('Low', axis=1, inplace=True)

	av_row = hist.mean(axis=1)
	return av_row.to_frame(stock)


@lru_cache(maxsize=100)
def _get_mcap(stock):
	print("Downloading mcap for", stock)
	ticker = yf.Ticker(stock)
	return str(ticker.info["marketCap"])


@app.route('/mcap')
def get_mcap():
	stock = request.args.get('stock', '')
	return _get_mcap(stock)



@app.route('/prices')
def get_prices():
	stock = request.args.get('stock', '')
	period = request.args.get('period', '')

	prices = get_history(stock, period)
	return prices.to_json()


@app.route('/last_price')
def get_last_price():
	stock = request.args.get('stock', '')
	prices = get_history(stock, '1w')

	return str(prices.tail(1).iloc[0][stock])
