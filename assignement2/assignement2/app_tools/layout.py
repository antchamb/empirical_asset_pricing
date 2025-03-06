
from app_tools.modal import *
from dash import html, dcc
import pandas as pd

alpha = pd.DataFrame([0, 0, 0, 0, 0], columns=["Col1"])
beta = pd.DataFrame([[0.5, 0], [0, 0.5], [0.5, 0.5], [0.3, 1.2], [0.7, 0.4]], columns=["Col1", "Col2"])
sigma = pd.DataFrame([[1, 0.5, 0.5, 0.5, 0.5],
                             [0.5, 1, 0.5, 0, 0],
                             [0.5, 0.5, 1, 0, 0],
                             [0.5, 0, 0, 1, 0.5],
                             [0.5, 0, 0, 0.5, 1]],
                             columns=[f"Var{i+1}" for i in range(5)])

mu_f = pd.DataFrame([[0.05], [0.07]], columns=["Value"])
sigma_f = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=["Col1", "Col2"])

parameters = html.Div([
    html.H3("Adjust Parameters:"),
    html.Div([
        dbc.Button("α", id='alpha', style={"marginTop": "10px", "width": "100%"}),
        dbc.Button("β", id="beta", style={"marginTop": "10px", "width": "100%"}),
        dbc.Button("Σ", id="sigma", style={"marginTop": "10px", "width": "100%"}),
        dbc.Button(html.Span(["μ", html.Sub("f")]), id="mu_f", style={"marginTop": "10px", "width": "100%"}),
        dbc.Button(html.Span(["Σ", html.Sub("f")]), id="sigma_f", style={"marginTop": "10px", "width": "100%"}),
    ]),
    create_matrix_modal("alpha", "α", "alpha-matrix", alpha),
    create_matrix_modal("beta", "β", "beta-matrix", beta),
    create_matrix_modal("sigma", "Σ", "sigma-matrix", sigma),
    create_matrix_modal("mu_f", "μ_f", "mu_f-matrix", mu_f),
    create_matrix_modal("sigma_f", "Σ_f", "sigma_f-matrix", sigma_f),
    html.Hr(),
    html.Div(
        [
            html.Label("T:"),
            dcc.Input(
                id="T",
                type="number",
                value=100,
                step=1,
                style={"marginBottom": "10px", "width": "100%"},
            )
        ]
    ),
    html.Hr(),
    html.Div(
        [
            html.Label("N:"),
            dcc.Input(
                id="N",
                type="number",
                value=1000,
                step=1,
                style={"marginBottom": "10px", "width": "100%"},
            ),
            dbc.Button(
                "Q3 & Q4 Simulation",
                id="sim-button",
                style={"marginTop": "10px", "width": "100%"},
            )
        ]
    ),

], style={
    "position": "fixed",
    "top": "0px",
    "left": "0px",
    "width": "7vw",
    "height": "100vh",
    "overflow-y": "auto",
    "backgroundColor": "#f8f9fa",
    "padding": "10px",
    "borderRight": "2px solid #ccc"
}
)

results = html.Div([
    html.H3("Question 1:"),
    html.Div([
        dcc.Graph(id="q1-plot-Re_t", style={"width": "50vw"}),
        dcc.Graph(id="q1-plot-ft", style={"width": "50vw"}),
    ],style={"display": 'flex', "flexDirection": "row"}
    ),
    html.Hr(),
    html.H3("Question 2:"),
    html.Div(id="q2-regression-results", style={"display": "flex", "flexDirection": "row"}),
    html.Hr(),
    html.H3("Question 3:"),
    html.Div([
        html.Div([
            dcc.Graph(id='q3-alpha-ols', style={"width": "50vw"}),
            dcc.Graph(id='q3-alpha-gls', style={"width": "50vw"}),
        ],style={"display": 'flex', "flexDirection": "row"}
        ),
        html.Div([
            dcc.Graph(id='q3-lambda-ols', style={"width": "50vw"}),
            dcc.Graph(id='q3-lambda-gls', style={"width": "50vw"}),
        ],style={"display": 'flex', "flexDirection": "row"}
        ),
    ]),
    html.Hr(),
    html.H3("Question 4:"),
    html.Div([
        html.Div([
            dcc.Graph(id='q4-alpha-ols', style={"width": "50vw"}),
            dcc.Graph(id='q4-alpha-gls', style={"width": "50vw"}),
        ],style={"display": 'flex', "flexDirection": "row"}
        ),
        html.Div([
            dcc.Graph(id='q4-lambda-ols', style={"width": "50vw"}),
            dcc.Graph(id='q4-lambda-gls', style={"width": "50vw"}),
        ],style={"display": 'flex', "flexDirection": "row"}
        ),
    ])
], style={
    "marginLeft": "8vw",
    "width": "90vw"
}
)

layout = html.Div(
    [parameters, results],
    style={"display": "flex", "flexDirection": "row", "width": "100%"},
)
