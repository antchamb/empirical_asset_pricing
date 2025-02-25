import dash
from dash import Input, Output, html, dcc
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
        ]
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
                "Run Simulation",
                id="run-sim-button",
                style={"marginTop": "10px", "width": "100%"},
            )
        ]
    ),
], style={
    "position": "fixed",  # ✅ Figer sur la page
    "top": "0px",
    "left": "0px",
    "width": "5vw",  # ✅ Largeur fixe
    "height": "100vh",  # ✅ Occupe toute la hauteur
    "overflow-y": "auto",  # ✅ Permet le scroll si besoin
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
    ],
    style={"width": "95vw", "padding": "20px", "marginLeft": "6vw"},
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



# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

