import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objs as go

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
def q3_q4(n_clicks, alpha, beta, sigma, mu_f, sigma_f, T, N):
    if None in locals().values():
        raise PreventUpdate

    alpha = np.array(pd.DataFrame(alpha).values, dtype=float)
    beta = np.array(pd.DataFrame(beta).values, dtype=float)
    sigma = np.array(pd.DataFrame(sigma).values, dtype=float)
    mu_f = np.array(pd.DataFrame(mu_f).values, dtype=float)
    sigma_f = np.array(pd.DataFrame(sigma_f).values, dtype=float)

    alpha_ols, lambda_ols, alpha_gls, lambda_gls = parameters_histogram(T, N, alpha, beta, sigma, mu_f, sigma_f)

    colors = ['rgba(255, 0, 0, 0.5)',
              'rgba(0, 255, 0, 0.5)',
              'rgba(0, 0, 255, 0.5)',
              'rgba(255, 165, 0, 0.5)',
              'rgba(128, 0, 128, 0.5)']

    a_ols_fig = go.Figure()
    for i in range(5):
        a_ols_fig.add_trace(go.Histogram(
            x=alpha_ols[:, i], name=f"Alpha Asset {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(5):
        a_ols_fig.add_trace(go.Scatter(
            x=[float(alpha[i]),float(alpha[i])], y=[0, 100],
            mode="lines", name=f"True a{i+1}", line=dict(color="black", dash="dash"),
        visible="legendonly"
        ))
    a_ols_fig.update_layout(
        title="Histograms of Alpha OLS",
        barmode='overlay',
        xaxis_title="Alpha Values",
        yaxis_title="Frequency",
        legend=dict(title="Assets")
    )


    a_gls_fig = go.Figure()
    for i in range(5):
        a_gls_fig.add_trace(go.Histogram(
            x=alpha_gls[:, i], name=f"Alpha Asset {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(5):
        a_gls_fig.add_trace(go.Scatter(
            x=[float(alpha[i]),float(alpha[i])], y=[0, 100],
            mode="lines", name=f"True {i + 1}", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    a_gls_fig.update_layout(
        title="Histograms of Alpha GLS",
        barmode='overlay',
        xaxis_title="Alpha Values",
        yaxis_title="Frequency",
        legend=dict(title="Assets")
    )

    l_ols_fig = go.Figure()
    for i in range(2):
        l_ols_fig.add_trace(go.Histogram(
            x=lambda_ols[:, i], name=f"Lambda {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(2):
        l_ols_fig.add_trace(go.Scatter(
            x=[float(mu_f[i]), float(mu_f[i])], y=[0, 100],
            mode="lines", name=f"True mu_f {i + 1}]", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    l_ols_fig.update_layout(
        title="Histograms of lambda OLS",
        barmode='overlay', xaxis_title="Lambda Values",
        yaxis_title="Frequency",
        legend=dict(title="Factors")
    )

    l_gls_fig = go.Figure()
    for i in range(2):
        l_gls_fig.add_trace(go.Histogram(
            x=lambda_gls[:, i], name=f"lambda {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(2):
        l_gls_fig.add_trace(go.Scatter(
            x=[float(mu_f[i]), float(mu_f[i])], y=[0, 100],
            mode="lines", name=f"True mu_f {i + 1}]", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    l_gls_fig.update_layout(
        title="Histograms of lambda GLS",
        barmode='overlay', xaxis_title="Lambda Values",
        yaxis_title="Frequency",
        legend=dict(title="Factors")
    )

    c_a_ols, c_l_ols, c_a_gls, c_l_gls = centered_estimators(alpha, mu_f, alpha_ols, lambda_ols, alpha_gls, lambda_gls, N)

    c_a_ols_fig = go.Figure()
    for i in range(5):
        c_a_ols_fig.add_trace(go.Histogram(
            x=c_a_ols[:, i], name=f"Alpha Asset {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(5):
        c_a_ols_fig.add_trace(go.Scatter(
            x=[float(alpha[i]), float(alpha[i])], y=[0, 100],
            mode="lines", name=f"True a{i + 1}", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    c_a_ols_fig.update_layout(
        title="Histograms of Centered Alpha OLS",
        barmode='overlay',
        xaxis_title="Alpha Values",
        yaxis_title="Frequency",
        legend=dict(title="Assets")
    )

    c_a_gls_fig = go.Figure()
    for i in range(5):
        c_a_gls_fig.add_trace(go.Histogram(
            x=c_a_gls[:, i], name=f"Alpha Asset {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(5):
        c_a_gls_fig.add_trace(go.Scatter(
            x=[float(alpha[i]), float(alpha[i])], y=[0, 100],
            mode="lines", name=f"True a{i + 1}", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    c_a_gls_fig.update_layout(
        title="Histograms of Centered Alpha GLS",
        barmode='overlay',
        xaxis_title="Alpha Values",
        yaxis_title="Frequency",
        legend=dict(title="Factors")
    )


    c_l_ols_fig = go.Figure()
    for i in range(2):
        c_l_ols_fig.add_trace(go.Histogram(
            x=c_l_ols[:, i], name=f"lambda {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(2):
        c_l_ols_fig.add_trace(go.Scatter(
            x=[float(mu_f[i]), float(mu_f[i])], y=[0, 100],
            mode="lines", name=f"True l{i + 1}", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    c_l_ols_fig.update_layout(
        title="Histograms of Centered lambda OLS",
        barmode='overlay',
        xaxis_title="Alpha Values",
        yaxis_title="Frequency",
        legend=dict(title="Factors")
    )


    c_l_gls_fig = go.Figure()
    for i in range(2):
        c_l_gls_fig.add_trace(go.Histogram(
            x=c_l_gls[:, i], name=f"lambda {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(2):
        c_l_gls_fig.add_trace(go.Scatter(
            x=[float(mu_f[i]), float(mu_f[i])], y=[0, 100],
            mode="lines", name=f"True l{i + 1}", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    c_l_gls_fig.update_layout(
        title="Histograms of Centered lambda GLS",
        barmode='overlay',
        xaxis_title="Alpha Values",
        yaxis_title="Frequency",
        legend=dict(title="Factors")
    )

    return a_ols_fig, l_ols_fig, a_gls_fig, l_gls_fig, c_a_ols_fig, c_l_ols_fig, c_a_gls_fig, c_l_gls_fig

@app.callback(
    [
        Output('q5-alpha-ols', 'figure'),
        Output('q5-lambda-ols', 'figure'),
        Output('q5-alpha-gls', 'figure'),
        Output('q5-lambda-gls', 'figure'),
    ],
    Input('q5-sim-button', 'n_clicks'),
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
def q5(n_clicks, alpha, beta, sigma, mu_f, sigma_f, T, N):
    if None in locals().values():
        raise PreventUpdate

    alpha = np.array(pd.DataFrame(alpha).values, dtype=float)
    beta = np.array(pd.DataFrame(beta).values, dtype=float)
    sigma = np.array(pd.DataFrame(sigma).values, dtype=float)
    mu_f = np.array(pd.DataFrame(mu_f).values, dtype=float)
    sigma_f = np.array(pd.DataFrame(sigma_f).values, dtype=float)

    alpha_ols, lambda_ols, alpha_gls, lambda_gls = estimate_q5_parameters(T, N, alpha, beta, sigma, mu_f, sigma_f)

    colors = ['rgba(255, 0, 0, 0.5)',
              'rgba(0, 255, 0, 0.5)',
              'rgba(0, 0, 255, 0.5)',
              'rgba(255, 165, 0, 0.5)',
              'rgba(128, 0, 128, 0.5)']
    a_ols_fig = go.Figure()
    for i in range(5):
        a_ols_fig.add_trace(go.Histogram(
            x=alpha_ols[:, i], name=f"Alpha Asset {i+1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(5):
        a_ols_fig.add_trace(go.Scatter(
            x=[np.mean(alpha_ols[:, i]), np.mean(alpha_ols[:, i])], y=[0, 100],
            mode="lines", name=f"E[a{i + 1}]", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    a_ols_fig.update_layout(
        title="Superimposed Histograms of Alpha OLS",
        barmode='overlay',  # Ensures histograms are overlaid
        xaxis_title="Alpha Values",
        yaxis_title="Frequency",
        legend=dict(title="Assets")
    )

    a_gls_fig = go.Figure()
    for i in range(5):
        a_gls_fig.add_trace(go.Histogram(
            x=alpha_gls[:, i], name=f"Alpha Asset {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(5):
        a_gls_fig.add_trace(go.Scatter(
            x=[np.mean(alpha_gls[:, i]), np.mean(alpha_gls[:, i])], y=[0, 100],
            mode="lines", name=f"E[a{i + 1}]", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    a_gls_fig.update_layout(
        title="Superimposed Histograms of Alpha GLS",
        barmode='overlay',
        xaxis_title="Alpha Values",
        yaxis_title="Frequency",
        legend=dict(title="Assets")
    )

    l_ols_fig = go.Figure()
    for i in range(2):
        l_ols_fig.add_trace(go.Histogram(
            x=lambda_ols[:, i], name=f"Lambda {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(2):
        l_ols_fig.add_trace(go.Scatter(
            x=[np.mean(lambda_ols[:, i]), np.mean(lambda_ols[:, i])], y=[0, 100],
            mode="lines", name=f"E[l{i + 1}]", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    l_ols_fig.update_layout(
        title="Superimposed Histograms of lambda OLS",
        barmode='overlay', xaxis_title="Lambda Values",
        yaxis_title="Frequency",
        legend=dict(title="Factors")
    )

    l_gls_fig = go.Figure()
    for i in range(2):
        l_gls_fig.add_trace(go.Histogram(
            x=lambda_gls[:, i], name=f"lambda {i + 1}",
            opacity=0.7,
            marker=dict(color=colors[i])
        ))
    for i in range(2):
        l_gls_fig.add_trace(go.Scatter(
            x=[np.mean(lambda_gls[:, i]), np.mean(lambda_gls[:, i])], y=[0, 100],
            mode="lines", name=f"E[l{i + 1}]", line=dict(color="black", dash="dash"),
            visible="legendonly"
        ))
    l_gls_fig.update_layout(
        title="Superimposed Histograms of lambda GLS",
        barmode='overlay', xaxis_title="Lambda Values",
        yaxis_title="Frequency",
        legend=dict(title="Factors")
    )


    return a_ols_fig, l_ols_fig, a_gls_fig, l_gls_fig

if __name__ == "__main__":
    app.run_server(debug=True)
