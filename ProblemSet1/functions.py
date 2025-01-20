import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Question 1
def routine(alpha, beta, theta, rho, var_u, var_v, cov_uv, T):
    errors = np.random.multivariate_normal(
        [0, 0],
        np.array([[var_u, cov_uv], [cov_uv, var_v]]),
        size=T,
    )
    u, v = errors[:, 0], errors[:, 1]

    x = np.zeros(T)
    x[0] = theta / (1 - rho)
    for t in range(1, T):
        x[t] = theta + rho * x[t - 1] + v[t]

    y = np.array([alpha + beta * x[t-1] + u[t] for t in range(1, T)])

    return x, y

# Question 2

def estimate_parameters(x, y):
    # OLS for y = alpha + beta * x
    X_y = add_constant(x[:-1])
    ols_y = OLS(y, X_y).fit()
    alpha_hat, beta_hat = ols_y.params
    r2_y = ols_y.rsquared

    # OLS for x = theta + rho * x
    X_x = add_constant(x[:-1])
    ols_x = OLS(x[1:], X_x).fit()
    theta_hat, rho_hat = ols_x.params
    r2_x = ols_x.rsquared

    residuals_u = y - ols_y.predict(X_y)
    residuals_v = x[1:] - ols_x.predict(X_x)
    var_u_hat = np.var(residuals_u)
    var_v_hat = np.var(residuals_v)
    cov_uv_hat = np.cov(residuals_u, residuals_v)[0, 1]

    return {
        "alpha": (alpha_hat, ols_y.pvalues[0], r2_y),
        "beta": (beta_hat, ols_y.pvalues[1], r2_y),
        "theta": (theta_hat, ols_x.pvalues[0], r2_x),
        "rho": (rho_hat, ols_x.pvalues[1], r2_x),
        "var_u": (var_u_hat, None, None),
        "var_v": (var_v_hat, None, None),
        "cov_uv": (cov_uv_hat, None, None),
    }
