# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:39:29 2021

@author: smong
"""

import numpy as np
import pandas as pd
import glob
from sklearn import __version__ as sklearn_version
import os

fn = 'Data_Points.csv'
try:
    os.remove(fn)
except:
    None

csv_files = glob.glob('.\\Tickers\\*.csv')
m = 400
i = 2
incr=500
steps = 18
h = True
for csv_name in csv_files[0:50]:
    ticker_df = pd.read_csv(csv_name)
    ticker_df = ticker_df[[x for x in ticker_df.columns[1:]]]
    for j in range(steps):
        close_diff = 2*(np.array(ticker_df['close'][1:-1]) - np.array(ticker_df['close'][0:-2]))/(np.array(ticker_df['close'][1:-1]) + np.array(ticker_df['close'][0:-2]))
        idx = m+1+j*incr
        subset = ticker_df[idx:(i+idx+1)]
        ticker_df_tosave = subset[:-1].reset_index(drop = True)
        
        for k in range(m):
            subset_close = np.array(ticker_df['close'][(idx-k-1):(i+idx-k-1)])
            subset_vol = np.array(ticker_df['volume'][(idx-k-1):(i+idx-k-1)])
            
            ticker_df_tosave[str('prev_' + str(k))] = subset_close/np.array(ticker_df_tosave['close'])
            ticker_df_tosave[str('prev_diff_' + str(k))] = close_diff[(idx-k-1):(i+idx-k-1)]
            ticker_df_tosave[str('prev_vol_' + str(k))] = subset_vol/np.array(ticker_df_tosave['volume'])
            
        close_diff = (np.array(subset['close'])[1:] - np.array(subset['close'])[:-1])
        ticker_df_tosave['next_day_change'] = np.array(np.sign(close_diff)*(abs(2*close_diff/(np.array(subset['close'])[1:] + np.array(subset['close'])[:-1]))>.01)).astype(int)
        closes = np.array(ticker_df.close[(idx):(i+idx+30)])
        close_diff = np.array([np.mean(closes[x:x+30]) - closes[x] for x in range(len(closes)-30)])
        ticker_df_tosave['next_month_change'] = close_diff/np.array(closes[:-30])
        ticker_df_tosave.to_csv(fn,mode = 'a', index = False, header = h)
        h = False