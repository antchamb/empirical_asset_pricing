import dash
from dash import Input, Output, State, html, dcc
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from functions import *


# Initialize the Dash app
app = dash.Dash(__name__)

# Layout
parameters = html.Div([
    html.H3("Adjust Parameters"),
    html.Div(
        [
            html.Label("α (alpha):"),
            dcc.Input(
                id="alpha",
                type="number",
                value=0.01,
                step=0.01,
                style={"marginBottom": "10px", "width": "100%"},
            ),
        ]
    ),
    html.Div(
        [
            html.Label("β (beta):"),
            dcc.Input(
                id="beta",
                type="number",
                value=0.05,
                step=0.01,
                style={"marginBottom": "10px", "width": "100%"},
            ),
        ]
    ),
    html.Div(
        [
            html.Label("θ (theta):"),
            dcc.Input(
                id="theta",
                type="number",
                value=0.01,
                step=0.01,
                style={"marginBottom": "10px", "width": "100%"},
            ),
        ]
    ),
    html.Div(
        [
            html.Label("ρ (rho):"),
            dcc.Input(
                id="rho",
                type="number",
                value=0.3,
                step=0.01,
                style={"marginBottom": "10px", "width": "100%"},
            ),
        ],
    ),
    html.Hr(),
    html.Div(
        [
            html.Label("σ²u (var_u):"),
            dcc.Input(
                id="var_u",
                type="number",
                value=0.6,
                step=0.01,
                style={"marginBottom": "10px", "width": "100%"},
            ),
        ]
    ),
    html.Div(
        [
            html.Label("σ²ν (var_v):"),
            dcc.Input(
                id="var_v",
                type="number",
                value=0.5,
                step=0.01,
                style={"marginBottom": "10px", "width": "100%"},
            ),
        ]
    ),
    html.Div(
        [
            html.Label("σuv (cov_uv):"),
            dcc.Input(
                id="cov_uv",
                type="number",
                value=-0.5,
                step=0.01,
                style={"marginBottom": "10px", "width": "100%"},
            ),
        ]
    ),
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
                value=10000,
                step=1,
                style={"marginBottom": "10px", "width": "100%"},
            ),
            dbc.Button(
                "Q3 Simulation",
                id="run-sim-button",
                style={"marginTop": "10px", "width": "100%"},
            )
        ]
    ),
    html.Hr(),
    html.Div([
        html.Label("N:"),
        dcc.Input(
            id="N-q4",
            type="number",
            value=100,
            step=1,
            style={"marginBottom": "10px", "width": "100%"},
        ),
        html.Label("T_min:"),
        dcc.Input(
            id="T-min-q4",
            type="number",
            value=50,
            step=1,
            style={"marginBottom": "10px", "width": "100%"},
        ),
        html.Label("T_max:"),
        dcc.Input(
            id="T-max-q4",
            type="number",
            value=1000,
            step=1,
            style={"marginBottom": "10px", "width": "100%"},
        ),
        html.Label("Segmentation:"),
        dcc.Input(
            id='T-parts',
            type="number",
            value=500,
            step=1,
            style={"marginBottom": "10px", "width": "100%"},
        ),
        dbc.Button(
            "Q4 & Q5 Simulation",
            id="q4-sim-button",
            style={"marginBottom": "10px", "width": "100%"},
        )
    ])
], style={
    "position": "fixed",
    "top": "0px",
    "left": "0px",
    "width": "5vw",
    "height": "100vh",
    "overflow-y": "auto",
    "backgroundColor": "#f8f9fa",
    "padding": "10px",
    "borderRight": "2px solid #ccc"
})


results = html.Div(
    [
        html.H3("Question 1:"),
        dcc.Graph(id="plot"),
        html.Hr(),
        html.H3("Question 2:"),
        html.Div(id="ols-results", style={"marginTop": "20px", "fontSize": "18px"}),
        html.Hr(),
        html.H3("Question 3:"),
        dcc.Graph(id="q3-plot"),
        html.Hr(),
        html.H3("Question 4:"),
        dcc.Graph(id="q4-plot"),
        html.Hr(),
        html.H3("Question 5:"),
        html.Div(id="ols-results-q5", style={"marginTop": "20px", "fontSize": "18px"}),
    ],
    style={"width": "90vw", "padding": "20px", "marginLeft": "6vw"},
)


app.layout = html.Div([parameters, results], style={'display': 'flex', 'flexDirection': 'inline-block'})

# Callback to update the graph based on parameters
@app.callback(
    [
        Output("plot", "figure"),
        Output("ols-results", "children")
    ],
    [
        Input("alpha", "value"),
        Input("beta", "value"),
        Input("theta", "value"),
        Input("rho", "value"),
        Input("var_u", "value"),
        Input("var_v", "value"),
        Input("cov_uv", "value"),
        Input("T", "value"),
    ],
)
def update_graph(alpha, beta, theta, rho, var_u, var_v, cov_uv, T):

    if None in locals().values():
        raise PreventUpdate

    # T = 100  # Fixed time series length
    x, y = routine(alpha, beta, theta, rho, var_u, var_v, cov_uv, T)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, T), y=x, mode="lines", name="x"))
    fig.add_trace(go.Scatter(x=np.arange(1, T), y=y, mode="lines", name="y"))
    fig.update_layout(
        title="Time Series of x and y",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white"
    )

    model_y, model_x = ols_regression(x, y)
    ols_results = html.Div([
        html.Pre(
            model_y.summary().as_text(),
            style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'backgroundColor': '#f9f9f9', 'padding': '10px'}
        ),

        html.Pre(model_x.summary().as_text(), style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'backgroundColor': '#f9f9f9', 'padding': '10px'}),
    ], style={
        "display": "flex",
        "flex-direction": "right",
        "width": "100vw"
    })
    return [fig, ols_results]

@app.callback(
    Output("q3-plot", "figure"),
    Input("run-sim-button", "n_clicks"),
    [
        State("alpha", "value"),
        State("beta", "value"),
        State("theta", "value"),
        State("rho", "value"),
        State("var_u", "value"),
        State("var_v", "value"),
        State("cov_uv", "value"),
        State("T", "value"),
        State("N", "value"),
    ]
)
def run_simulation(n_clicks, alpha, beta, theta, rho, var_u, var_v, cov_uv, T, N):
    if None in locals().values():
        raise PreventUpdate
    T, N = int(T), int(N)

    beta_estimates = beta_repartition(alpha, beta, theta, rho, var_u, var_v, cov_uv, T, N)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=beta_estimates, nbinsx=50, name="Estimated β"))

    fig.add_trace(go.Scatter(
        x=[beta, beta], y=[0, 500],
        mode="lines", name="β",
        line=dict(color="green", dash="dash")
    ))

    beta_hat_mean = np.mean(beta_estimates)

    fig.add_trace(go.Scatter(
        x=[beta_hat_mean, beta_hat_mean], y=[0, 500],
        mode="lines", name="E[β̂]",
        line=dict(color="red", dash="dash")
    ))

    fig.update_layout(
        title="Distribution of Estimated β (Bias Visualization)",
        xaxis_title="Estimated β",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    return fig

@app.callback(
    [
       Output('q4-plot', 'figure'),
        Output("ols-results-q5", "children")
    ],
    Input("q4-sim-button", "n_clicks"),
    [
        State("alpha", "value"),
        State("beta", "value"),
        State("theta", "value"),
        State("rho", "value"),
        State("var_u", "value"),
        State("var_v", "value"),
        State("cov_uv", "value"),
        State("N-q4", "value"),
        State("T-min-q4", "value"),
        State("T-max-q4", "value"),
        State("T-parts", "value"),
    ]
)
def q4_q5_sim(n_clicks, alpha, beta, theta, rho, var_u, var_v, cov_uv, N, Tmin, Tmax, Tparts):
    if None in [alpha, beta, theta, rho, var_u, var_v, cov_uv, N, Tmin, Tmax, Tparts]:
        raise PreventUpdate

    Tvalues = np.round(np.linspace(Tmin, Tmax, Tparts)).astype(int)
    Tvalues = np.unique(Tvalues)

    beta_biases = []

    for t in Tvalues:
        beta_estimates = beta_repartition(alpha, beta, theta, rho, var_u, var_v, cov_uv, t, N)
        bias = np.mean(beta_estimates) - beta
        beta_biases.append(bias)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Tvalues, y=beta_biases, mode="lines+markers", name="Bias of β̂",
        line=dict(color="blue")
    ))

    fig.update_layout(
        title="Bias of Estimated β̂ vs. Sample Size T",
        xaxis_title="Sample Size (T)",
        yaxis_title="Bias (E[β̂] - β)",
        template="plotly_white"
    )

    model = ols_bias(beta_biases, Tvalues)
    ols_result = html.Div([
        html.Pre(model.summary().as_text(),
                 style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'backgroundColor': '#f9f9f9',
                        'padding': '10px'}),
    ], style={
        "display": "flex",
        "justify-content": "center",
        "width": "100vw"
    })

    return [fig, ols_result]


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

