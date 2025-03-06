import numpy as np
import statsmodels.api as sm
from dash import html


def simulate_factor_model(T, alpha, beta, sigma, mu_f, sigma_f):

    mu_f = np.ravel(mu_f)
    # print("mu_f shape", mu_f.shape)
    # print("sigma_f shape", sigma_f.shape)
    ft = np.random.multivariate_normal(mu_f, sigma_f, T)
    errors = np.random.multivariate_normal(np.zeros(5), sigma, T)
    errors += alpha.T
    Re_t = np.dot(ft, beta.T) + errors

    return Re_t, ft


def estimates_model_parameters(Re_t, beta, sigma):

    ols_model = sm.OLS(
        np.mean(Re_t, axis=0),
        sm.add_constant(beta)
    ).fit()
    ols_summary = ols_model.summary()

    gls_model = sm.GLS(
        np.mean(Re_t, axis=0),
        sm.add_constant(beta),
        sigma=sigma
    ).fit()
    gls_summary = gls_model.summary()

    regressions_results = html.Div([
        html.Pre(
            ols_summary.as_text(),
            style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'backgroundColor': '#f9f9f9', 'padding': '10px'}
        ),
        html.Pre(
            gls_summary.as_text(),
            style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'backgroundColor': '#f9f9f9', 'padding': '10px'}
        ),
    ], style={
    "display": "flex",
    "flexDirection": "row",
    "justifyContent": "center",
    "alignItems": "center",
    "width": "100vw"
    })

    return regressions_results

def parameters_histogram(T, N, alpha, beta, sigma, mu_f, sigma_f):

    results = {"ols": {"alpha":[], "lambda":[]}, "gls": {"alpha":[], "lambda":[]}}
    for i in range(1, N+1):
        Re_t, ft = simulate_factor_model(T, alpha, beta, sigma, mu_f, sigma_f)
        ols_model = sm.OLS(
            np.mean(Re_t, axis=0),
            sm.add_constant(beta)
        ).fit()

        gls_model = sm.GLS(
            np.mean(Re_t, axis=0),
            sm.add_constant(beta),
            sigma=sigma
        ).fit()

        results["ols"]["alpha"].append(ols_model.params[0])
        results["ols"]["lambda"].append([ols_model.params[1], ols_model.params[2]])

        results['gls']['alpha'].append(gls_model.params[0])
        results['gls']['lambda'].append([gls_model.params[1], gls_model.params[2]])

    alpha_hat_ols = np.array(results['ols']['alpha'])
    lambda_hat_ols = np.array(results['ols']['lambda'])
    # print(alpha_hat_ols.shape)
    alpha_hat_gls = np.array(results['gls']['alpha'])
    lambda_hat_gls = np.array(results['gls']['lambda'])

    return alpha_hat_ols, lambda_hat_ols, alpha_hat_gls, lambda_hat_gls

def centered_estimators(alpha, mu_f, alpha_ols, lambda_ols, alpha_gls, lambda_gls):

    center_a_ols = (alpha_ols - np.mean(alpha)) / np.std(alpha_ols)
    center_l1_ols = (lambda_ols[:, 0] - mu_f[0]) / np.std(lambda_ols[:, 0])
    center_l2_ols = (lambda_ols[:, 1] - mu_f[1]) / np.std(lambda_ols[:, 1])

    center_a_gls = (alpha_gls - np.mean(alpha)) / np.std(alpha_ols)
    center_l1_gls = (lambda_gls[:, 0] - mu_f[0]) / np.std(lambda_gls[:, 0])
    center_l2_gls = (lambda_gls[:, 1] - mu_f[1]) / np.std(lambda_gls[:, 1])
    print(center_a_ols)
    return center_a_ols, center_l1_ols, center_l2_ols, center_a_gls, center_l1_gls, center_l2_gls

