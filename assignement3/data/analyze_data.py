# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:43:52 2025

@author: antch
"""

import pandas as pd
import datetime as dt
import numpy as np
from scipy.optimize import minimize

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

# df['utility'] = (df['C'] ** (1 - gamma) * df['C'].shift(1) ** (-k * (1 - gamma))) / (1 - gamma)
# df['M'] = beta * (df['C'] / df['C'].shift(1)) ** (k * (gamma - 1)) * \
#     (df['C'].shift(-1) / df['C']) ** (1 - gamma)

# df['m'] = np.log(df['M'])

# df = df.iloc[1:-1, :]


def computeG(theta, df):
    """
        Computes the moment conditions using the given instruments Z_T
    """
    gamma, delta, k = theta
    
    df['Clag'] = df['C'].shift(1)
    df['M'] = delta * (df['C'] / df['C'].shift(1)) ** (k * (gamma - 1)) * \
         (df['C'].shift(-1) / df['C']) ** (1 - gamma)
    df['M'] = df['M'].shift(1)
    df.dropna(inplace=True)
    
    gt = np.column_stack([df['M'] * df['Mkt'] - 1] * 4)
    zt = np.column_stack((np.ones(len(gt)), df['C'], df['Clag'], df['Mkt']))
    
    G = gt * zt
    g_mean = np.mean(G, axis=0)
    
    return g_mean.T @ g_mean

theta = [0.99, 2, 3]
# G = computeG(theta, df)

def gmmFirstStage(theta, df):
    
    gamma, delta, k = theta
    
    df['Clag'] = df['C'].shift(1)
    df['M'] = delta * (df['C'] / df['C'].shift(1)) ** (k * (gamma - 1)) * \
         (df['C'].shift(-1) / df['C']) ** (1 - gamma)
    df['M'] = df['M'].shift(1)
    df.dropna(inplace=True)
    
    def computeG(df):
        """
            Computes the moment conditions using the given instruments Z_T
        """

        
        gt = np.column_stack([df['M'] * df['Mkt'] - 1] * 4)
        zt = np.column_stack((np.ones(len(gt)), df['C'], df['Clag'], df['Mkt']))
        
        G = gt * zt
        g_mean = np.mean(G, axis=0)
        
        return g_mean.T @ g_mean
    
    results = minimize(computeG, theta, method='Nelder-Mead')
    
    return results.x    

theta_est = gmmFirstStage(theta, df)



# def neweyWestKernelEstimator(G):
    
#     #  Truncation parameter
#     m = int(4 * ((G.shape[0] / 100) ** (2/9)))
#     T, k = G.shape
#     S_hat = (G.T @ G) / T
    
#     for j in range(1, m+1):
#         Gammaj = (G[:-j].T @ G[j:]) / T
#         S_hat += (1 - j / (m + 1)) * (Gammaj + Gammaj.T)

#     return S_hat

# S_hat = neweyWestKernelEstimator(G)


    
