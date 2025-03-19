# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:43:52 2025

@author: antch
"""

import pandas as pd
import datetime as dt

ff_df = pd.read_csv(r'FF_data.csv', skiprows=3)
ff_df = ff_df.iloc[:-1, :]
ff_df['Date'] = ff_df['Date'].astype(str)
ff_df['Date'] = pd.to_datetime(ff_df['Date'], format='%Y%m')
ff_df['Date'] = ff_df['Date'].dt.strftime('%Y-%m-%d')

fred_df = pd.read_csv(r'FRED_data.csv')

