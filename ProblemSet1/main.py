import numpy as np


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

routine(0.01, 0.05, 0.01, 0.3, 0.6, 0.5, -0.5, 100)
