import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm

# Question 1
def routine(alpha, beta, theta, rho, var_u, var_v, cov_uv, T):
    # Generate correlated normal random variables
    errors = np.random.multivariate_normal(
        [0, 0],
        np.array([[var_u, cov_uv], [cov_uv, var_v]]),
        size=T
    )
    u, v = errors[:, 0], errors[:, 1]

    # Initialize x, y
    x, y = np.zeros(T), np.zeros(T)
    x[0] = theta / (1 - rho)  # x0 depends only on theta and rho
    y[0] = alpha + beta * x[0]
    for t in range(1, T):
        x[t] = theta + rho * x[t - 1] + v[t]
        y[t] = alpha + beta * x[t - 1] + u[t]

    return x, y

# # Question 2
#
def estimate_parameters(x, y):
    """ Estimate parameters for the two predictive regressions using OLS. """

    # OLS for y_t+1 = α + β * x_t + u_t+1
    X_y = sm.add_constant(x[:-1])
    ols_y = sm.OLS(y[1:], X_y).fit()

    # OLS for x_t+1 = θ + ρ * x_t + ν_t+1
    X_x = sm.add_constant(x[:-1])
    ols_x = sm.OLS(x[1:], X_x).fit()

    # Extract summary tables from statsmodels
    table_y = ols_y.summary2().tables[1]
    table_x = ols_x.summary2().tables[1]

    return table_y, table_x