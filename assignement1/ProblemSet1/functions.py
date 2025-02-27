import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm

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


# Question 2
def ols_regression(x, y):
    # Regression 1: y[t+1] = alpha + beta * x[t] + u[t+1]
    Y = sm.add_constant(x[:-1])
    y_target = y[1:]
    model_y = OLS(y_target, Y).fit()

    X = sm.add_constant(x[:-1])
    x_target = x[1:]
    model_x = OLS(x_target, X).fit()

    return model_y, model_x

# Question 3 & 4:
def beta_repartition(alpha, beta, theta, rho, var_u, var_v, cov_uv, T, N):
    x, y = np.zeros((T, N)), np.zeros((T, N))

    errors = np.random.multivariate_normal(
        [0, 0],
        np.array([[var_u, cov_uv], [cov_uv, var_v]]),
        size=(T, N)
    )

    u, v = errors[:, :, 0], errors[:, :, 1]

    x[0, :] = theta / (1 - rho)
    y[0, :] = alpha + beta * x[0, :]

    for t in range(1, T):
        x[t, :] = theta + rho * x[t-1, :] + v[t, :]
        y[t, :] = alpha + beta * x[t - 1, :] + u[t, :]

    beta_estimates = np.zeros(N)

    for i in range(N):
        X = sm.add_constant(x[:-1, i])
        y_target = y[1:, i]
        model = sm.OLS(y_target, X).fit()
        beta_estimates[i] = model.params[1]

    return beta_estimates

# Question 5:
def ols_bias(biases, Tvalues):
    Tvalues = np.array(Tvalues)

    X = np.column_stack([np.ones(len(Tvalues)), 1/Tvalues, 1/(Tvalues**2)])
    y = np.array(biases)

    model = sm.OLS(y, X).fit()

    return model
