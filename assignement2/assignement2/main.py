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
    ],
    [
        Input("q3-sim-button", "n_clicks"),
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
    alpha_ols_fig.update_layout(
        title="α OLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )

    lambda_ols_fig = go.Figure()
    lambda_ols_fig.add_trace(go.Histogram(x=lambda_ols[:, 0], name="λ_hat_1"))
    lambda_ols_fig.add_trace(go.Histogram(x=lambda_ols[:, 1], name="λ_hat_2"))
    lambda_ols_fig.update_layout(
        title="λ OLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )


    alpha_gls_fig = go.Figure()
    alpha_gls_fig.add_trace(go.Histogram(x=alpha_gls, name="α_hat"))
    alpha_gls_fig.update_layout(
        title="α GLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )

    lambda_gls_fig = go.Figure()
    lambda_gls_fig.add_trace(go.Histogram(x=lambda_gls[:, 0], name="λ_hat_1"))
    lambda_gls_fig.add_trace(go.Histogram(x=lambda_gls[:, 1], name="λ_hat_2"))
    lambda_gls_fig.update_layout(
        title="λ GLS",
        title_x=0.5,
        xaxis_title="Values",
        yaxis_title="Frequency",
    )


    return alpha_ols_fig, lambda_ols_fig, alpha_gls_fig, lambda_gls_fig


if __name__ == "__main__":
    app.run_server(debug=True)
