# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:43:52 2025

@author: antch
"""

import pandas as pd
import datetime as dt
import numpy as np

ff_df = pd.read_csv(r'FF_data.csv', skiprows=3).iloc[:-1,:]
ff_df["Date"] = pd.to_datetime(ff_df["Date"].astype(str), format="%Y%m")
ff_df.set_index('Date', inplace=True)

ff_df['Mkt'] = 1 + (ff_df['Mkt-RF'] + ff_df['RF']) / 100
ff_df['RF'] = 1 + ff_df['RF'] / 100

ff = ff_df[['Mkt', 'RF']].resample("Q").prod() - 1
ff.index = ff.index.to_period("Q").to_timestamp(how='start')


fred_df = pd.read_csv("FRED_data.csv", parse_dates=["observation_date"])
fred_df.rename(columns={"observation_date": "Date", "PCECC96": "C"}, inplace=True)
fred_df.set_index("Date", inplace=True)

df = fred_df.merge(ff, left_index=True, right_index=True, how='left')

k = 3
beta = 2
gamma = 0.5

df['utility'] = (df['C'] ** (1 - gamma) * df['C'].shift(1) ** (-k * (1 - gamma))) / (1 - gamma)
df['M'] = beta * (df['C'] / df['C'].shift(1)) ** (k * (gamma - 1)) * \
    (df['C'].shift(-1) / df['C']) ** (1 - gamma)

df['m'] = np.log(df['M'])

df = df.iloc[1:-1, :]

