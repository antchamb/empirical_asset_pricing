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
        print(Re_t.shape)
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

    alpha_hat_ols = np.array([results['ols']['alpha']])
    lambda_hat_ols = np.array(results['ols']['lambda'])
    # print(alpha_hat_ols.shape)
    alpha_hat_gls = np.array([results['gls']['alpha']])
    lambda_hat_gls = np.array(results['gls']['lambda'])
    # print(lambda_hat_ols.shape)
    return alpha_hat_ols, lambda_hat_ols, alpha_hat_gls, lambda_hat_gls
