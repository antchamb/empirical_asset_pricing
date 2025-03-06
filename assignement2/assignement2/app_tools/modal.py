import dash_bootstrap_components as dbc
from dash import dash_table

def create_matrix_modal(button_id, title, table_id, df):
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(f"Modify {title} Matrix")),
            dbc.ModalBody(
                dash_table.DataTable(
                    id=table_id,
                    columns=[{"name": col, "id": col, "type": "numeric"} for col in df.columns],
                    data=df.to_dict("records"),
                    editable=True,
                    style_table={'overflowX': 'auto'}
                )
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id=f"close_{button_id}", className="ms-auto", n_clicks=0)
            ),
        ],
        id=f"modal_{button_id}",
        is_open=False,
    )
