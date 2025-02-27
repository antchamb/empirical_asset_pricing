# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:30:10 2025

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

a = np.zeros(5)

beta = np.array([
    [0.5, 0],
    [0, 0.5],
    [0.5, 0.5],
    [0.3, 1.2],
    [0.7, 0.4]
])

sigma = np.array([
    [1, 0.5, 0.5, 0.5, 0.5],
    [0.5, 1, 0.5, 0, 0],
    [0.5, 0.5, 1, 0, 0],
    [0.5, 0, 0, 1, 0.5],
    [0.5, 0, 0, 0.5, 1]
])

mu_f = np.array([0.05, 0.07])

sigma_f = np.array([
    [1, 0.5],
    [0.5, 1]
])


def simulate_factor_model(T, a, beta, sigma, mu_f, sigma_f):
    
    ft = np.random.multivariate_normal(mu_f, sigma_f, T)
    
    errors = np.random.multivariate_normal(np.zeros(5), sigma)
    
    Re_t = a + np.dot(ft, beta.T) + errors
    
    return Re_t, ft

Re_t, ft = simulate_factor_model(100, a, beta, sigma, mu_f, sigma_f)

plt.plot(np.arange(1,101), Re_t)
plt.show()

def estimate_parameters(Re_t, beta, sigma):
    
    ols_model = sm.OLS(
        np.mean(Re_t, axis=0),
        sm.add_constant(beta)
        ).fit()
    gls_model = sm.GLS(
        np.mean(Re_t, axis=0),
        sm.add_constant(beta),
        sigma=sigma
        ).fit()
    
    ols_summary = ols_model.summary()
    gls_summary = gls_model.summary()
    
    print("\nOLS Model Summary:\n", ols_summary)
    print("\nGLS Model Summary:\n", gls_summary)
    

estimate_parameters(Re_t, beta, sigma)
