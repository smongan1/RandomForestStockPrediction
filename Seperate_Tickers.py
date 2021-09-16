# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:24:52 2021

@author: smong
"""

import pandas as pd


csv_path = '.\\historical_stock_prices.csv'

stocks = pd.read_csv(csv_path)[['ticker', 'date', 'close','volume']]
ticker_1980 = stocks[stocks['date']<'1980']['ticker'].unique()

tickers = pd.DataFrame()
tickers['ticker'] = ticker_1980

stocks = stocks[(stocks['date'] < '2018') & (stocks['date'] > '1980')]

dates = pd.DataFrame()
dates['date'] = stocks['date'].unique()

for x in range(len(ticker_1980)):
    stock_subset = stocks[stocks.ticker == tickers.values[x][0]].sort_values('date').reset_index(drop = True)
    stock_subset = stock_subset.merge(dates, on = 'date', how = 'right')[['date','close','volume']].sort_values('date').reset_index(drop = True)
    stock_subset.interpolate(method = 'nearest',inplace = True)
    stock_subset.fillna(stock_subset.min(), inplace = True)
    stock_subset.to_csv(path_or_buf = '.\\Tickers\\' + tickers.values[x][0] + '.csv')