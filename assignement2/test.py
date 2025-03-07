# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 13:30:09 2025

@author: dell
"""

import numpy as np
import statsmodels.api as sm
from scipy.linalg import cholesky, solve_triangular

T = 10
N = 100
alpha = np.array([0,0,0,0,0])
beta = np.array([[0.5, 0], [0, 0.5], [0.5, 0.5], [0.3, 1.2], [0.7, 0.4]])
sigma = np.array([[1, 0.5, 0.5, 0.5, 0.5],
                             [0.5, 1, 0.5, 0, 0],
                             [0.5, 0.5, 1, 0, 0],
                             [0.5, 0, 0, 1, 0.5],
                             [0.5, 0, 0, 0.5, 1]])
mu_f = np.array([[0.05], [0.07]])
sigma_f = np.array([[1.0, 0.5], [0.5, 1.0]])

def simulate_factor_model(T, alpha, beta, sigma, mu_f, sigma_f):

    mu_f = np.ravel(mu_f)
    # print("mu_f shape", mu_f.shape)
    # print("sigma_f shape", sigma_f.shape)
    ft = np.random.multivariate_normal(mu_f, sigma_f, T)
    errors = np.random.multivariate_normal(np.zeros(5), sigma, T)
    errors += alpha.T
    Re_t = np.dot(ft, beta.T) + errors

    return Re_t, ft



results = {
    "ols": {
        "beta": np.zeros((N+1, 5, 2)),
        "alpha": [],
        "lambda": []
        },
    "parameters":{
        "sigma_hat": np.zeros((N+1, 5, 5)),
        "sigma_f_hat": np.zeros((N+1, 2, 2)),
        "errors": np.zeros((N+1, T, T)),
        "ft": np.zeros((N+1, T, 2)),
        "Re_t": np.zeros((N+1, T, 5))
        }
    }

omega_hat = np.zeros((N+1, 5, T+1))



for _ in range(1, N+1):
    
    mu_f = np.ravel(mu_f)
    ft = np.random.multivariate_normal(mu_f, sigma_f, T)
    errors = np.random.multivariate_normal(np.zeros(5), sigma, T)
    
    results["parameters"]["errors"][_] = np.dot(errors, errors.T)
    test = np.dot(errors, errors.T)
    sigma_hat = np.cov(errors.T, bias=False)
    sigma_f_hat = np.cov(ft.T, bias=False)
    
    errors += alpha.T
    Re_t = np.dot(ft, beta.T) + errors
    
    
    beta_hat_ols = np.zeros((5,2))
    for i in range(5):
        
        ols_model = sm.OLS(Re_t[:, i], sm.add_constant(ft)).fit()
        beta_hat_ols[i, :] = ols_model.params[1:]
    
    ols_lambda = np.linalg.inv(beta_hat_ols.T @ beta_hat_ols) @ beta_hat_ols.T @ np.mean(Re_t, axis=0).reshape(-1,1)
    ols_alpha = np.mean(Re_t, axis=0).reshape(-1,1) - np.dot(beta_hat_ols, ols_lambda).reshape(-1,1)
    ols_alpha = ols_alpha.reshape(-1)
    results["ols"]["beta"][_] = beta_hat_ols
    results["ols"]["alpha"].append(ols_alpha)
    results["ols"]["lambda"].append(ols_lambda)
    results["parameters"]["sigma_hat"][_] = sigma_hat
    results["parameters"]["sigma_f_hat"][_] = sigma_f_hat
    results["parameters"]["ft"][_] = ft
    results["parameters"]["Re_t"][_] = Re_t
    

omega_hat = np.mean(results["parameters"]["errors"], axis=0)
omega_inv = np.linalg.inv(omega_hat)

gls_results = {
    "beta": np.zeros((N+1, 5, 2)),
    "lambda": [],
    "alpha": []}

for _ in range(1, N+1):
    ft = results["parameters"]["ft"][_]
    Re_t = results["parameters"]["Re_t"][_]
    sigma = results["parameters"]["sigma_hat"][_]
    sigma_f = results["parameters"]["sigma_f_hat"][_]
    
    
    X = sm.add_constant(ft)
    Xomega = np.dot(X.T, omega_inv)
    
    beta_gls = np.dot(
        np.dot(np.linalg.inv(np.dot(Xomega, X)), Xomega),
        Re_t)
    beta_gls = beta_gls[1:,:]
    beta_gls = beta_gls.T
    
    gls_results["beta"][_] = beta_gls
    
    lambda_gls = np.linalg.inv(beta_gls.T @ np.linalg.inv(sigma) @ beta) @ beta_gls.T @ np.linalg.inv(sigma) @ np.mean(Re_t, axis=0)
    gls_results["lambda"].append(lambda_gls)
    
    
    alpha_gls = np.mean(Re_t, axis=0) - beta_gls @ lambda_gls
    gls_results["alpha"].append(alpha_gls)
    

# Ebeta_gls = np.mean(gls_results["beta"], axis=0)
# Ebeta_ols = np.mean(results["ols"]["beta"], axis=0)
# E_lambda_gls = np.mean(np.array(gls_results["lambda"]), axis=0)
# E_alpha_gls = np.array(gls_results["alpha"])
alpha_gls = np.array(gls_results["alpha"])
lambda_gls = np.array(gls_results["lambda"])
alpha_ols = np.array(results["ols"]["alpha"])
lambda_ols = np.array(results["ols"]["lambda"])



print(np.mean(results["ols"]["beta"], axis=0))
# print(np.mean(results["gls"]["beta"], axis=0))
print(np.mean(gls_results["beta"], axis=0))