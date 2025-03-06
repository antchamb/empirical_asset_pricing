# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:30:10 2025

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from collections import defaultdict


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


# Question 1
def simulate_factor_model(T, a, beta, sigma, mu_f, sigma_f):
    
    ft = np.random.multivariate_normal(mu_f, sigma_f, T)
    
    errors = np.random.multivariate_normal(np.zeros(5), sigma)
    
    Re_t = a + np.dot(ft, beta.T) + errors
    
    return Re_t, ft

Re_t, ft = simulate_factor_model(1000, a, beta, sigma, mu_f, sigma_f)

plt.plot(Re_t)
# plt.plot(np.arange(1,101), Re_t)
# plt.show()

# Question 2
# def estimate_parameters(Re_t, beta, sigma):
    
#     ols_model = sm.OLS(
#         np.mean(Re_t, axis=0),
#         sm.add_constant(beta)
#         ).fit()
#     gls_model = sm.GLS(
#         np.mean(Re_t, axis=0),
#         sm.add_constant(beta),
#         sigma=sigma
#         ).fit()
    
#     # ols_summary = ols_model.summary()
#     # gls_summary = gls_model.summary()
    
    
#     alpha_ols = ols_model.params[0]
#     lambda_ols = np.array([ols_model.params[1], ols_model.params[2]])
    
#     alpha_gls = gls_model.params[0]
#     lambda_gsl = np.array(gls_model.params[1:])
    
#     return alpha_ols, lambda_ols, alpha_gls, lambda_gsl
    
    

# # estimate_parameters(Re_t, beta, sigma)

# # Lambda =  np.dot(
# #     np.dot(np.linalg.inv(np.dot(beta.T,beta)), beta.T),
# #     np.mean(Re_t, axis=0)
# # )

# # alpha = np.mean(Re_t, axis=0) - np.dot(beta, Lambda)
# # print("lambda = ", Lambda)
# # print("\nalpha = ", alpha)


# # Question 3
# results = {"ols": {"alpha": [], "lambda": []}, "gls": {"alpha": [], "lambda": []}}


# for i in range(1, 1001):
#     Re_t, ft = simulate_factor_model(10, a, beta, sigma, mu_f, sigma_f)
#     alpha_ols, lambda_ols, alpha_gls, lambda_gls = estimate_parameters(Re_t, beta, sigma)
#     results['ols']['alpha'].append(alpha_ols)
#     results['ols']['lambda'].append(lambda_ols)
#     results['gls']['alpha'].append(alpha_gls)
#     results['gls']['lambda'].append(lambda_gls)
    
# alpha_ols_values = np.array(results['ols']['alpha'])
# lambda_ols_values = np.array(results['ols']['lambda'])

# alpha_gls_values = np.array(results['gls']['alpha'])
# lambda_gls_values = np.array(results['gls']['lambda'])

# E_alpha_hat_ols = np.mean(alpha_ols_values)
# E_lambda_hat_ols = np.mean(lambda_ols_values, axis=0)

# E_alpha_hat_gls = np.mean(alpha_gls_values)
# E_lambda_hat_gls = np.mean(lambda_gls_values, axis=0)

# fig, axes = plt.subplots(3, 2, figsize=(12, 12))


# E_alpha_hat_ols = np.mean(alpha_ols_values)
# E_lambda_hat_ols = np.mean(lambda_ols_values, axis=0)

# E_alpha_hat_gls = np.mean(alpha_gls_values)
# E_lambda_hat_gls = np.mean(lambda_gls_values, axis=0)


# fig, axes = plt.subplots(3, 2, figsize=(12, 12))


# axes[0, 0].hist(alpha_ols_values, bins=30, alpha=0.7, label="OLS", density=True)
# axes[0, 0].axvline(x=0, color='g', linestyle="--", label="True Alpha")
# axes[0, 0].axvline(x=E_alpha_hat_ols, color='r', linestyle="--", label="Estimated Alpha")
# axes[0, 0].set_title("Distribution of Alpha (OLS)")
# axes[0, 0].legend()

# axes[0, 1].hist(alpha_gls_values, bins=30, alpha=0.7, label="GLS", density=True)
# axes[0, 1].axvline(x=0, color='g', linestyle="--", label="True Alpha")
# axes[0, 1].axvline(x=E_alpha_hat_gls, color='r', linestyle="--", label="Estimated Alpha")
# axes[0, 1].set_title("Distribution of Alpha (GLS)")
# axes[0, 1].legend()

# axes[1, 0].hist(lambda_ols_values[:, 0], bins=30, alpha=0.7, label="OLS", density=True)
# axes[1, 0].axvline(x=mu_f[0], color='g', linestyle="--", label="True Lambda_1")
# axes[1, 0].axvline(x=E_lambda_hat_ols[0], color='r', linestyle="--", label="Estimated Lambda_1")
# axes[1, 0].set_title("Distribution of Lambda_1 (OLS)")
# axes[1, 0].legend()

# axes[1, 1].hist(lambda_gls_values[:, 0], bins=30, alpha=0.7, label="GLS", density=True)
# axes[1, 1].axvline(x=mu_f[0], color='g', linestyle="--", label="True Lambda_1")
# axes[1, 1].axvline(x=E_lambda_hat_gls[0], color='r', linestyle="--", label="Estimated Lambda_1")
# axes[1, 1].set_title("Distribution of Lambda_1 (GLS)")
# axes[1, 1].legend()

# axes[2, 0].hist(lambda_ols_values[:, 1], bins=30, alpha=0.7, label="OLS", density=True)
# axes[2, 0].axvline(x=mu_f[1], color='g', linestyle="--", label="True Lambda_2")
# axes[2, 0].axvline(x=E_lambda_hat_ols[1], color='r', linestyle="--", label="Estimated Lambda_2")
# axes[2, 0].set_title("Distribution of Lambda_2 (OLS)")
# axes[2, 0].legend()

# axes[2, 1].hist(lambda_gls_values[:, 1], bins=30, alpha=0.7, label="GLS", density=True)
# axes[2, 1].axvline(x=mu_f[1], color='g', linestyle="--", label="True Lambda_2")
# axes[2, 1].axvline(x=E_lambda_hat_gls[1], color='r', linestyle="--", label="Estimated Lambda_2")
# axes[2, 1].set_title("Distribution of Lambda_2 (GLS)")
# axes[2, 1].legend()

# # Define true values of theta
# theta_alpha = 0  # True value of alpha
# theta_lambda = mu_f  # True values of lambda


# # Convert to NumPy arrays
# alpha_ols_values = np.array(results['ols']['alpha'])
# lambda_ols_values = np.array(results['ols']['lambda'])
# se_alpha_ols_values = np.array(results['ols']['se_alpha'])
# se_lambda_ols_values = np.array(results['ols']['se_lambda'])

# alpha_gls_values = np.array(results['gls']['alpha'])
# lambda_gls_values = np.array(results['gls']['lambda'])
# se_alpha_gls_values = np.array(results['gls']['se_alpha'])
# se_lambda_gls_values = np.array(results['gls']['se_lambda'])

# # Compute standardized ratios
# alpha_ols_ratios = (alpha_ols_values - 0) / se_alpha_ols_values
# lambda_ols_ratios = (lambda_ols_values - mu_f) / se_lambda_ols_values

# alpha_gls_ratios = (alpha_gls_values - 0) / se_alpha_gls_values
# lambda_gls_ratios = (lambda_gls_values - mu_f) / se_lambda_gls_values

# # Compute standardized ratios using true values
# alpha_ols_ratios = (alpha_ols_values - theta_alpha) / se_alpha_ols_values
# lambda_ols_ratios = (lambda_ols_values - theta_lambda) / se_lambda_ols_values

# alpha_gls_ratios = (alpha_gls_values - theta_alpha) / se_alpha_gls_values
# lambda_gls_ratios = (lambda_gls_values - theta_lambda) / se_lambda_gls_values

# # Plot the distributions of the ratios
# fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# axes[0, 0].hist(alpha_ols_ratios, bins=30, alpha=0.7, label="OLS", density=True)
# axes[0, 0].set_title("Standardized Ratio of Alpha (OLS)")
# axes[0, 0].legend()

# axes[0, 1].hist(alpha_gls_ratios, bins=30, alpha=0.7, label="GLS", density=True)
# axes[0, 1].set_title("Standardized Ratio of Alpha (GLS)")
# axes[0, 1].legend()

# axes[1, 0].hist(lambda_ols_ratios[:, 0], bins=30, alpha=0.7, label="OLS", density=True)
# axes[1, 0].set_title("Standardized Ratio of Lambda_1 (OLS)")
# axes[1, 0].legend()

# axes[1, 1].hist(lambda_gls_ratios[:, 0], bins=30, alpha=0.7, label="GLS", density=True)
# axes[1, 1].set_title("Standardized Ratio of Lambda_1 (GLS)")
# axes[1, 1].legend()

# axes[2, 0].hist(lambda_ols_ratios[:, 1], bins=30, alpha=0.7, label="OLS", density=True)
# axes[2, 0].set_title("Standardized Ratio of Lambda_2 (OLS)")
# axes[2, 0].legend()

# axes[2, 1].hist(lambda_gls_ratios[:, 1], bins=30, alpha=0.7, label="GLS", density=True)
# axes[2, 1].set_title("Standardized Ratio of Lambda_2 (GLS)")
# axes[2, 1].legend()

# plt.show()

