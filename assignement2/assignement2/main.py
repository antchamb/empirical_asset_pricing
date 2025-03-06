import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.python.ops.numpy_ops.np_dtypes import float64

from app_tools.layout import layout
from functions import *

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = layout

# CALLBACKS
@app.callback(
    Output("modal_alpha", "is_open"),
    Input("alpha", "n_clicks"),
    Input("close_alpha", "n_clicks"),
    State("modal_alpha", "is_open"),
)
def toggle_modal_alpha(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal_beta", "is_open"),
    Input("beta", "n_clicks"),
    Input("close_beta", "n_clicks"),
    State("modal_beta", "is_open"),
)
def toggle_modal_beta(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal_sigma", "is_open"),
    Input("sigma", "n_clicks"),
    Input("close_sigma", "n_clicks"),
    State("modal_sigma", "is_open"),
)
def toggle_modal_sigma(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal_mu_f", "is_open"),
    Input("mu_f", "n_clicks"),
    Input("close_mu_f", "n_clicks"),
    State("modal_mu_f", "is_open"),
)
def toggle_modal_mu_f(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal_sigma_f", "is_open"),
    Input("sigma_f", "n_clicks"),
    Input("close_sigma_f", "n_clicks"),
    State("modal_sigma_f", "is_open"),
)
def toggle_modal_sigma_f(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    [
        Output("q1-plot-Re_t", "figure"),
        Output("q1-plot-ft", "figure"),
        Output("q2-regression-results", "children"),
    ],
    [
        Input("alpha-matrix", "data"),
        Input("beta-matrix", "data"),
        Input("sigma-matrix", "data"),
        Input("mu_f-matrix", "data"),
        Input("sigma_f-matrix", "data"),
        Input("T", "value"),
    ]
)
def q1_q2(alpha, beta, sigma, mu_f, sigma_f, T):
    if None in locals().values():
        raise PreventUpdate

    alpha = np.array(pd.DataFrame(alpha).values, dtype=float)
    beta = np.array(pd.DataFrame(beta).values, dtype=float)
    sigma = np.array(pd.DataFrame(sigma).values, dtype=float)
    mu_f = np.array(pd.DataFrame(mu_f).values, dtype=float)
    sigma_f = np.array(pd.DataFrame(sigma_f).values, dtype=float)

    Re_t, ft = simulate_factor_model(T, alpha, beta, sigma, mu_f, sigma_f)

    fig_Re_t = go.Figure()
    for i in range(Re_t.shape[1]):
        fig_Re_t.add_trace(go.Scatter(y=Re_t[:, i], mode="lines", name=f"Asset {i+1}"))
    fig_Re_t.update_layout(title="Simulated Assets", xaxis_title="Time", yaxis_title="Excess Return")

    fig_ft = go.Figure()
    for i in range(ft.shape[1]):
        fig_ft.add_trace(go.Scatter(y=ft[:, i], mode="lines", name=f"Factor {i+1}"))

    regressions_results = estimates_model_parameters(Re_t, beta, sigma)

    return fig_Re_t, fig_ft, regressions_results

@app.callback(
    [
        Output("q3-alpha-ols", "figure"),
        Output("q3-lambda-ols", "figure"),
        Output("q3-alpha-gls", "figure"),
        Output("q3-lambda-gls", "figure"),
        Output("q4-alpha-ols", "figure"),
        Output("q4-lambda-ols", "figure"),
        Output("q4-alpha-gls", "figure"),
        Output("q4-lambda-gls", "figure"),
    ],
    [
        Input("sim-button", "n_clicks"),
    ],
    [
        State("alpha-matrix", "data"),
        State("beta-matrix", "data"),
        State("sigma-matrix", "data"),
        State("mu_f-matrix", "data"),
        State("sigma_f-matrix", "data"),
        State("T", "value"),
        State("N", "value")
    ]
)
def q3(n_clicks, alpha, beta, sigma, mu_f, sigma_f, T, N):
    if None in locals().values():
        raise PreventUpdate

    alpha = np.array(pd.DataFrame(alpha).values, dtype=float)
    beta = np.array(pd.DataFrame(beta).values, dtype=float)
    sigma = np.array(pd.DataFrame(sigma).values, dtype=float)
    mu_f = np.array(pd.DataFrame(mu_f).values, dtype=float)
    sigma_f = np.array(pd.DataFrame(sigma_f).values, dtype=float)

    alpha_ols, lambda_ols, alpha_gls, lambda_gls = parameters_histogram(T, N, alpha, beta, sigma, mu_f, sigma_f)

    alpha_ols_fig = go.Figure()
    alpha_ols_fig.add_trace(go.Histogram(x=alpha_ols, name="α_hat"))
    alpha_ols_fig.add_trace(go.Scatter(
        x=[np.mean(alpha), np.mean(alpha)], y=[0, max(alpha_ols)],
        mode="lines", name="E[a]",
    ))
    alpha_ols_fig.update_layout(
        title="alpha OLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )

    lambda_ols_fig = go.Figure()
    lambda_ols_fig.add_trace(go.Histogram(x=lambda_ols[:, 0], name="λ_hat_1"))
    lambda_ols_fig.add_trace(go.Histogram(x=lambda_ols[:, 1], name="λ_hat_2"))
    lambda_ols_fig.add_trace(go.Scatter(
        x=[float(mu_f[0]), float(mu_f[0])], y=[0, 100],
        mode="lines", name="True λ1",
        line=dict(color="green", dash="dash"),
    ))
    lambda_ols_fig.add_trace(go.Scatter(
        x=[float(mu_f[1]), float(mu_f[1])], y=[0, 100],
        mode="lines", name="True λ2",
        line=dict(color="purple", dash="dash"),
    ))
    lambda_ols_fig.update_layout(
        title="lambda OLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )


    alpha_gls_fig = go.Figure()
    alpha_gls_fig.add_trace(go.Histogram(x=alpha_gls, name="α_hat"))
    alpha_gls_fig.add_trace(go.Scatter(
        x=[np.mean(alpha), np.mean(alpha)], y=[0, 100],
        mode="lines", name="E[a]",
    ))
    alpha_gls_fig.update_layout(
        title="alpha GLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )

    lambda_gls_fig = go.Figure()
    lambda_gls_fig.add_trace(go.Histogram(x=lambda_gls[:, 0], name="λ_hat_1"))
    lambda_gls_fig.add_trace(go.Histogram(x=lambda_gls[:, 1], name="λ_hat_2"))
    lambda_gls_fig.add_trace(go.Scatter(
        x=[float(mu_f[0]), float(mu_f[0])], y=[0, 100],
        mode="lines", name="True λ1",
        line=dict(color="green", dash="dash"),
    ))
    lambda_gls_fig.add_trace(go.Scatter(
        x=[float(mu_f[1]), float(mu_f[1])], y=[0, 100],
        mode="lines", name="True λ2",
        line=dict(color="purple", dash="dash"),
    ))
    lambda_gls_fig.update_layout(
        title="lambda GLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )

    centered_a_ols, centered_l1_ols, centered_l2_ols, centered_a_gls, centered_l1_gls, centered_l2_gls = centered_estimators(alpha, mu_f, alpha_ols, lambda_ols, alpha_gls, lambda_gls)

    center_a_fig = go.Figure()
    center_a_fig.add_trace(go.Histogram(x=centered_a_ols, name="c_alpha_hat"))
    center_a_fig.update_layout(
        title="centered alpha OLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )

    center_l_fig = go.Figure()
    center_l_fig.add_trace(go.Histogram(x=centered_l1_ols, name="c_lambda1_hat"))
    center_l_fig.add_trace(go.Histogram(x=centered_l2_ols, name="c_lambda2_hat"))
    center_l_fig.update_layout(
        title="centered lambda OLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )

    center_a_gls_fig = go.Figure()
    center_a_gls_fig.add_trace(go.Histogram(x=centered_a_gls, name="c_alpha_hat"))
    center_a_gls_fig.update_layout(
        title="centered alpha GLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )

    center_l_gls_fig = go.Figure()
    center_l_gls_fig.add_trace(go.Histogram(x=centered_l1_gls, name="c_lambda1_hat"))
    center_l_gls_fig.add_trace(go.Histogram(x=centered_l2_gls, name="c_lambda2_hat"))
    center_l_gls_fig.update_layout(
        title="centered lambda GLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )

    print(centered_a_ols.shape)
    print(alpha_ols.shape)
    return alpha_ols_fig, lambda_ols_fig, alpha_gls_fig, lambda_gls_fig, center_a_fig, center_l_fig, center_a_gls_fig, center_l_gls_fig


if __name__ == "__main__":
    app.run_server(debug=True)
