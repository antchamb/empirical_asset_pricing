import dash
import dash_table
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
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

    if None in locals().values():
        raise PreventUpdate

    T = 100  # Fixed time series length
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

    table_y, table_x = estimate_parameters(x, y)

    # Convert tables to Dash DataTable
    def create_dash_table(df, title):
        return html.Div([
            html.H4(title, style={"textAlign": "center", "marginTop": "20px"}),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict("records"),
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "padding": "5px"},
                style_header={"fontWeight": "bold", "backgroundColor": "lightgrey"},
            )
        ])
    return fig, html.Div([
        create_dash_table(table_y, "Regression 1: y_t+1 = α + β * x_t + u_t+1"),
        create_dash_table(table_x, "Regression 2: x_t+1 = θ + ρ * x_t + ν_t+1")
    ])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
