import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from functions import *


# Initialize the Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div(
    style={"display": "flex", "height": "100vh"},
    children=[
        # Parameter controls
        html.Div(
            style={
                "width": "5vw",
                "padding": "20px",
                "backgroundColor": "#f9f9f9",
                "borderRight": "1px solid #ddd",
            },
            children=[
                html.H3("Adjust Parameters"),
                html.Div(
                    style={"marginTop": "20px"},
                    children=[
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
                    ],
                ),
            ],
        ),
        # Graph
        html.Div(
            style={"width": "95vw", "padding": "20px"},
            children=[
                html.H3("Plot of x and y"),
                dcc.Graph(id="plot"),
                html.H3("Estimated Parameters"),
                html.Div(id="ols-results", style={"marginTop": "20px", "fontSize": "18px"}),
            ],
        ),
    ],
)

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
    ],
)
def update_graph(alpha, beta, theta, rho, var_u, var_v, cov_uv):
    T = 100  # Fixed time series length
    x, y = routine(alpha, beta, theta, rho, var_u, var_v, cov_uv, T)

    figure = {
        "data": [
            go.Scatter(x=np.arange(1, T), y=x[1:], mode="lines", name="x"),
            go.Scatter(x=np.arange(1, T), y=y, mode="lines", name="y"),
        ],
        "layout": go.Layout(
            title="Time Series of x and y",
            xaxis={"title": "Time"},
            yaxis={"title": "Value"},
            template="plotly_white",
        ),
    }

    params = estimate_parameters(x, y)
    # Generate statistical boxes
    stats_boxes = []
    for param, (value, p_value, r2) in params.items():
        box_content = [html.P(f"{param}: {value:.4f}")]
        if p_value is not None:
            box_content.append(html.P(f"p-value: {p_value:.4f}"))
        if r2 is not None:
            box_content.append(html.P(f"R²: {r2:.4f}"))
        stats_boxes.append(
            html.Div(
                box_content,
                style={
                    "border": "1px solid #ddd",
                    "borderRadius": "5px",
                    "padding": "10px",
                    "margin": "5px",
                    "width": "200px",
                    "backgroundColor": "#f9f9f9",
                },
            )
        )
    return figure, stats_boxes

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
