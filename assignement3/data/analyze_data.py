# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:43:52 2025

@author: antch
"""

import pandas as pd
import datetime as dt
import numpy as np
from scipy.optimize import minimize

# load and process FF data
ff_df = pd.read_csv(r'FF_data.csv', skiprows=3).iloc[:-1,:]
ff_df["Date"] = pd.to_datetime(ff_df["Date"].astype(str), format="%Y%m")
ff_df.set_index('Date', inplace=True)
ff_df['Mkt'] = 1 + (ff_df['Mkt-RF'] + ff_df['RF']) / 100
ff_df['RF'] = 1 + ff_df['RF'] / 100
ff = ff_df[['Mkt', 'RF']].resample("Q").prod() - 1
ff.index = ff.index.to_period("Q").to_timestamp(how='start')

# Load and process FRED data
fred_df = pd.read_csv("FRED_data.csv", parse_dates=["observation_date"])
fred_df.rename(columns={"observation_date": "Date", "PCECC96": "C"}, inplace=True)
fred_df.set_index("Date", inplace=True)
# Takin comsumption growth for realistic output
fred_df['C'] = fred_df['C'].pct_change()
fred_df.dropna(inplace=True)
df = fred_df.merge(ff, left_index=True, right_index=True, how='left')


# df['utility'] = (df['C'] ** (1 - gamma) * df['C'].shift(1) ** (-k * (1 - gamma))) / (1 - gamma)
# df['M'] = beta * (df['C'] / df['C'].shift(1)) ** (k * (gamma - 1)) * \
#     (df['C'].shift(-1) / df['C']) ** (1 - gamma)
# df['m'] = np.log(df['M'])
# df = df.iloc[1:-1, :]

# =============================================================================
#                           FUNCTIONS
# =============================================================================

def computeResidual(theta, df):
    """
    Computes the moment conditions using the given instruments Z_T.
    """
    # unpack parameters
    gamma, delta, k = theta
    
    df = df.copy()
    
    # lag consumption growth
    df['Clag'] = df['C'].shift(1)
    
    # Compute SDF
    df['M'] = delta * (df['C'] / df['C'].shift(1)) ** (k * (gamma - 1)) * \
              (df['C'].shift(-1) / df['C']) ** (1 - gamma)
    df['M'] = df['M'].shift(1)
    df.dropna(inplace=True)
    
    # Compute moment conditions
    # gt = np.column_stack([df['M'] * df['Mkt'] - 1] * 4)
    # zt = np.column_stack((np.ones(len(gt)), df['C'], df['Clag'], df['Mkt']))
    gt = (df['M'] * df['Mkt'] - 1).values[:, np.newaxis]  # shape (T,1)
    zt = np.column_stack((np.ones(len(gt)), df['C'], df['Clag'], df['Mkt']))  # shape (T,4)

    G = gt * zt
    
    return G

# first stage GMM estimation function
def gmmFirstStage(df, theta_initial):
    
    def objectiveFunc(theta):
        
        G = computeResidual(theta, df)
        G_mean = np.mean(G, axis=0)
        
        return G_mean.T @ G_mean
    
    results = minimize(objectiveFunc, theta_initial, method='Nelder-Mead')

    return results.x  # Return estimated parameters


# Function to estimate Newey-West covariance matrix for GMM moments
def neweyWestKernelEstimator(G):
    
    #  Truncation parameter
    m = int(4 * ((G.shape[0] / 100) ** (2/9)))
    T, k = G.shape
    S_hat = (G.T @ G) / T
    
    for j in range(1, m+1):
        Gammaj = (G[:-j].T @ G[j:]) / T
        S_hat += (1 - j / (m + 1)) * (Gammaj + Gammaj.T)

    return S_hat

# Second-stage GMM estimation with optimal weighting matrix
def gmmSecondStage(df, theta_hat, S_hat):
    
    W_T = np.linalg.inv(S_hat)
    
    def objectiveFunc(theta):
        
        G = computeResidual(theta, df)
        G_mean = np.mean(G, axis=0)
        
        return G_mean.T @ W_T @ G_mean
    
    results = minimize(objectiveFunc, theta_hat, method='Nelder-Mead')
    
    return results.x


# =============================================================================
#                       RESULTS
# =============================================================================

# initial theta
theta = [2, 0.99, 0.5]


theta_hat = gmmFirstStage(df, theta)

# Compute the covariance matrix using residuals from first stage
S_hat = neweyWestKernelEstimator(computeResidual(theta_hat, df))


theta_hat_2 = gmmSecondStage(df, theta_hat, S_hat)

print('\ngamma estimation: ', theta_hat_2[0])
print('This parameter represents the investor relative risk aversion. The estimate is economically reasonable. It suggests moderate risk aversion\n')
print('delta estimation: ', theta_hat_2[1])
print('represent how investors discount future utility. The estimation suggests severe instability\n')
print('kappa estimation: ', theta_hat_2[2])
print('reflects how strongly consumption habits affect utility')


