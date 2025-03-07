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
    results = {"ols": {"alpha": [], "lambda": []}, "gls": {"alpha": [], "lambda": []}}
    for i in range(1, N + 1):
        Re_t, ft = simulate_factor_model(T, alpha, beta, sigma, mu_f, sigma_f)
        ols_model = sm.OLS(
            np.mean(Re_t, axis=0),
            beta
        ).fit()

        gls_model = sm.GLS(
            np.mean(Re_t, axis=0),
            beta,
            sigma=sigma
        ).fit()

        lambda_ols = [ols_model.params[0], ols_model.params[1]]
        alpha_ols = np.mean(Re_t, axis=0) - np.dot(beta, np.array(lambda_ols))

        results["ols"]["alpha"].append(alpha_ols)
        results["ols"]["lambda"].append(lambda_ols)

        lambda_gls = [gls_model.params[0], gls_model.params[1]]
        alpha_gls = np.mean(Re_t, axis=0) - np.dot(beta, np.array(lambda_gls))

        results['gls']['alpha'].append(alpha_gls)
        results['gls']['lambda'].append(lambda_gls)

    alpha_hat_ols = np.array(results['ols']['alpha'])
    lambda_hat_ols = np.array(results['ols']['lambda'])
    alpha_hat_gls = np.array(results['gls']['alpha'])
    lambda_hat_gls = np.array(results['gls']['lambda'])

    return alpha_hat_ols, lambda_hat_ols, alpha_hat_gls, lambda_hat_gls


def centered_estimators(alpha, mu_f, alpha_ols, lambda_ols, alpha_gls, lambda_gls, N):
    alpha = alpha.reshape(-1)
    c_a_ols = alpha_ols - alpha
    c_l_ols = lambda_ols - mu_f.T
    c_a_gls = alpha_gls - alpha
    c_l_gls = lambda_gls - mu_f.T

    c_a_ols /= np.std(alpha_ols, axis=0)
    c_l_ols /= np.std(lambda_ols, axis=0)
    c_a_gls /= np.std(alpha_gls, axis=0)
    c_l_gls /= np.std(lambda_gls, axis=0)

    return c_a_ols, c_l_ols, c_a_gls, c_l_gls


def estimate_q5_parameters(T, N, alpha, beta, sigma, mu_f, sigma_f):
    results = {
        "ols": {
            "beta": np.zeros((N + 1, 5, 2)),
            "alpha": [],
            "lambda": []
        },
        "parameters": {
            "sigma_hat": np.zeros((N + 1, 5, 5)),
            "sigma_f_hat": np.zeros((N + 1, 2, 2)),
            "errors": np.zeros((N + 1, T, T)),
            "ft": np.zeros((N + 1, T, 2)),
            "Re_t": np.zeros((N + 1, T, 5))
        }
    }

    omega_hat = np.zeros((N + 1, 5, T + 1))

    for _ in range(1, N + 1):

        mu_f = np.ravel(mu_f)
        ft = np.random.multivariate_normal(mu_f, sigma_f, T)
        errors = np.random.multivariate_normal(np.zeros(5), sigma, T)

        results["parameters"]["errors"][_] = np.dot(errors, errors.T)
        test = np.dot(errors, errors.T)
        sigma_hat = np.cov(errors.T, bias=False)
        sigma_f_hat = np.cov(ft.T, bias=False)

        errors += alpha.T
        Re_t = np.dot(ft, beta.T) + errors

        beta_hat_ols = np.zeros((5, 2))
        for i in range(5):
            ols_model = sm.OLS(Re_t[:, i], sm.add_constant(ft)).fit()
            beta_hat_ols[i, :] = ols_model.params[1:]

        ols_lambda = np.linalg.inv(beta_hat_ols.T @ beta_hat_ols) @ beta_hat_ols.T @ np.mean(Re_t, axis=0).reshape(-1,
                                                                                                                   1)
        ols_alpha = np.mean(Re_t, axis=0).reshape(-1, 1) - np.dot(beta_hat_ols, ols_lambda).reshape(-1, 1)
        ols_alpha = ols_alpha.reshape(-1)
        ols_lambda = ols_lambda.reshape(-1)
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
        "beta": np.zeros((N + 1, 5, 2)),
        "lambda": [],
        "alpha": []}

    for _ in range(1, N + 1):
        ft = results["parameters"]["ft"][_]
        Re_t = results["parameters"]["Re_t"][_]
        sigma = results["parameters"]["sigma_hat"][_]
        sigma_f = results["parameters"]["sigma_f_hat"][_]

        X = sm.add_constant(ft)
        Xomega = np.dot(X.T, omega_inv)

        beta_gls = np.dot(
            np.dot(np.linalg.inv(np.dot(Xomega, X)), Xomega),
            Re_t)
        beta_gls = beta_gls[1:, :]
        beta_gls = beta_gls.T

        gls_results["beta"][_] = beta_gls

        lambda_gls = np.linalg.inv(beta_gls.T @ np.linalg.inv(sigma) @ beta) @ beta_gls.T @ np.linalg.inv(
            sigma) @ np.mean(Re_t, axis=0)
        gls_results["lambda"].append(lambda_gls)

        alpha_gls = np.mean(Re_t, axis=0) - beta_gls @ lambda_gls
        gls_results["alpha"].append(alpha_gls)

    alpha_gls = np.array(gls_results["alpha"])
    lambda_gls = np.array(gls_results["lambda"])
    alpha_ols = np.array(results["ols"]["alpha"])
    lambda_ols = np.array(results["ols"]["lambda"])
    return alpha_ols, lambda_ols, alpha_gls, lambda_gls


