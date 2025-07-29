# !pip install ubiops dash pandas plotly seaborn
# pip install dash-mantine-components
import ubiops
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from dateutil import parser

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


import dash_mantine_components as dmc
from dash import ctx 

# ==============================================================================
# 1. UbiOps API Configuration & Helper Functions
# ==============================================================================

# --- Store credentials securely. Deel niet met derden onbewust---
API_TOKENS = {
    "poc": "Token 481b206efdcee52b165f011605263baea8d6319a", # REPLACE with your POC token
    "chat": "Token 9155a7c8fef85ff102bf6c3dddf6b3deb51a7586"  # REPLACE with your Chat token
}
# Example label to filter metrics for a specific deployment version
GEMMA_DEPLOYMENT_LABEL_POC = "deployment_version_id:07736fa1-9999-44c1-9dbd-7de83f43663f" # REPLACE if needed

def make_connection(project_name):
    """Establishes a connection to the UbiOps API for a given project."""
    configuration = ubiops.Configuration()
    configuration.api_key['Authorization'] = API_TOKENS[project_name]
    configuration.host = "https://api.demo.vlam.ai/v2.1" # Adjust if necessary
    api_client = ubiops.ApiClient(configuration)
    return ubiops.CoreApi(api_client)

def get_time_series_metric(api, project_name, metric_name, start_date, end_date, aggregation_s, labels=None):
    """Fetches aggregated time-series data from UbiOps."""
    try:
        response = api.time_series_data_list(
            project_name=project_name,
            metric=metric_name,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            aggregation_period=aggregation_s,
            labels=labels
        )
        data_points = response.to_dict().get('data_points', [])
        if not data_points:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['timestamp', 'value'])
        df = pd.DataFrame([
            {"timestamp": pd.to_datetime(dp["start_date"]), "value": dp["value"]}
            for dp in data_points
        ])
        return df
    except ubiops.exceptions.ApiException as e:
        print(f"API Error fetching '{metric_name}': {e}")
        return pd.DataFrame(columns=['timestamp', 'value'])

def get_slow_requests_percentage(api, project_name, start_date, end_date, threshold_s=3.0):
    """
    Calculates the percentage of requests that took longer than a given threshold.
    NOTE: This fetches individual request logs, which can be slow for large time ranges.
    """
    try:
        # This endpoint fetches individual request details
        all_requests = api.deployment_requests_list(
            project_name=project_name,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            status='completed',
            deployment_name='vlam-chat-gemma'
        )

        
        if not all_requests:
            return 0.0

        # durations = [req['duration'] for req in all_requests if req['duration'] is not None]
        durations = []
        for req in all_requests:
            # If req is a dict
            if isinstance(req, dict):
                duration = req.get('duration')
            else:
                # If req is an object, try attribute access
                duration = getattr(req, 'duration', None)
            if duration is not None:
                durations.append(duration)
        if not durations:
            return 0.0
            
        slow_requests = [d for d in durations if d > threshold_s]
        percentage = (len(slow_requests) / len(durations)) * 100
        return percentage

    except ubiops.exceptions.ApiException as e:
        print(f"API Error fetching deployment requests: {e}")
        return "Error"
    except Exception as ex:
        print(f"An error occurred in get_slow_requests_percentage: {ex}")
        return "Error"

def get_slow_requests_percentage_from_df(df, threshold_s=3.0):
    """
    Calculates the percentage of requests where time to first token > threshold_s.
    """
    if df.empty or 'value' not in df.columns:
        return 0.0
    slow_count = (df['value'] > threshold_s).sum()
    total_count = len(df)
    if total_count == 0:
        return 0.0
    return (slow_count / total_count) * 100


# ==============================================================================
# 2. Dash Application Layout
# ==============================================================================

app = dash.Dash(__name__)
# app = JupyterDash(__name__)

app.layout = dmc.MantineProvider(
    html.Div(
        style={
            'fontFamily': 'Segoe UI, Arial, sans-serif',
            'background': 'linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%)',
            'minHeight': '100vh',
            'padding': '0',
            'margin': '0',
        },
        children=[
            # --- Header Bar ---
            html.Div(style={
                'background': 'linear-gradient(90deg, #2563eb 0%, #38bdf8 100%)',
                'padding': '24px 40px 18px 40px',
                'color': 'white',
                'display': 'flex',
                'alignItems': 'center',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.07)',
                'borderBottomLeftRadius': '18px',
                'borderBottomRightRadius': '18px',
                'marginBottom': '32px',
            }, children=[
                html.Div("🚀", style={'fontSize': '2.5rem', 'marginRight': '18px'}),
                html.H1("UbiOps Server Usage Dashboard", style={'margin': 0, 'fontWeight': 700, 'fontSize': '2.2rem', 'letterSpacing': '1px'}),
                html.Div(style={'flex': 1}),
                html.A("by Your Team", href="#", style={'color': 'white', 'fontWeight': 300, 'fontSize': '1rem', 'textDecoration': 'none', 'marginLeft': '20px'})
            ]),

            # --- Main Content ---
            html.Div(style={
                'display': 'flex',
                'gap': '32px',
                'padding': '0 40px 40px 40px',
                'alignItems': 'flex-start',
                'maxWidth': '1600px',
                'margin': '0 auto',
            }, children=[
                # --- Controls Sidebar ---
                dmc.Paper(
                    shadow="md",
                    radius="lg",
                    p="xl",
                    style={
                        'minWidth': '320px',
                        'maxWidth': '360px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'gap': '28px',
                        'position': 'sticky',
                        'top': '32px',
                        'background': 'white',
                    },
                    children=[
                        dmc.Title("Controls", order=3, style={'color': '#2563eb', 'marginBottom': 0}),
                        dmc.Divider(variant="dashed"),
                        dmc.Stack(
                            gap="xs",
                            children=[
                                dmc.Text("Select Project:", fw=500, fz="sm"),
                                dmc.Select(
                                    id='project-selector',
                                    data=[{'label': 'poc (Gemma)', 'value': 'poc'}, {'label': 'chat (alles)', 'value': 'chat'}],
                                    value='poc',
                                    style={'marginBottom': 8},
                                    size="md",
                                    radius="md"
                                ),
                            ]
                        ),
                        dmc.Divider(variant="dashed"),
                        dmc.Stack(
                            gap="xs",
                            children=[
                                dmc.Text("Select Start Date & Time:", fw=500, fz="sm"),
                                dmc.DateTimePicker(
                                    id='start-datetime-picker',
                                    value=(datetime.datetime.now() - datetime.timedelta(days=1)),
                                    style={'marginBottom': 8},
                                    size="md",
                                    radius="md",
                                    clearable=True,
                                    dropdownType="modal"
                                ),
                            ]
                        ),
                        dmc.Stack(
                            gap="xs",
                            children=[
                                dmc.Text("Select End Date & Time:", fw=500, fz="sm"),
                                dmc.DateTimePicker(
                                    id='end-datetime-picker',
                                    value=datetime.datetime.now(),
                                    style={'marginBottom': 8},
                                    size="md",
                                    radius="md",
                                    clearable=True,
                                    dropdownType="modal"
                                ),
                            ]
                        ),
                        dmc.Divider(variant="dashed"),
                        dmc.Stack(
                            gap="xs",
                            children=[
                                dmc.Text("Aggregation (seconds):", fw=500, fz="sm"),
                                dmc.NumberInput(
                                    id='aggregation-input',
                                    value=3600,
                                    min=60,
                                    step=60,
                                    size="md",
                                    radius="md",
                                    style={'marginBottom': 8}
                                ),
                            ]
                        ),
                        dmc.Divider(variant="dashed"),
                        dmc.Stack(
                            gap="xs",
                            children=[
                                dmc.Text("Select Metrics to Plot:", fw=500, fz="sm"),
                                dmc.Group(
                                    gap="xs",
                                    children=[
                                        dmc.Button("Select All", id="select-all-metrics", size="xs", color="blue", variant="light"),
                                        dmc.Button("Deselect All", id="deselect-all-metrics", size="xs", color="gray", variant="light"),
                                    ],
                                    style={'marginBottom': 6}
                                ),
                                dmc.CheckboxGroup(
                                    id='metric-selector',
                                    value=['deployments.requests'],
                                    size="md",
                                    children=[
                                        dmc.Checkbox(label='Requests (request/m)', value='deployments.requests'),
                                        dmc.Checkbox(label='Failed Requests', value='deployments.failed_requests'),
                                        dmc.Checkbox(label='Request Duration', value='deployments.request_duration'),
                                        dmc.Checkbox(label='Input Volume', value='deployments.input_volume'),
                                        dmc.Checkbox(label='Output Volume', value='deployments.output_volume'),
                                        dmc.Checkbox(label='Express Queue Time', value='deployments.express_queue_time'),
                                        dmc.Checkbox(label='Batch Queue Time', value='deployments.batch_queue_time'),
                                        dmc.Checkbox(label='Credits', value='deployments.credits'),
                                        # dmc.Checkbox(label='Instances', value='deployments.instances'),
                                        # dmc.Checkbox(label='Express Queue Size', value='deployments.express_queue_size'),
                                        # dmc.Checkbox(label='Batch Queue Size', value='deployments.batch_queue_size'),
                                        dmc.Checkbox(label='Network In', value='deployments.network_in'),
                                        dmc.Checkbox(label='Network Out', value='deployments.network_out'),
                                        # dmc.Checkbox(label='Instance Start Time', value='deployments.instance_start_time'),
                                        # dmc.Checkbox(label='Token Count', value='custom.token_count'),
                                        dmc.Checkbox(label='Total Tokens', value='custom.total_tokens'),
                                        dmc.Checkbox(label='Prompt Tokens', value='custom.prompt_tokens'),
                                        dmc.Checkbox(label='Completion Tokens', value='custom.completion_tokens'),
                                        dmc.Checkbox(label='Completion Tokens Cumulative', value='custom.completion_tokens_cumulative'),
                                        dmc.Checkbox(label='Prompt Tokens Cumulative', value='custom.prompt_tokens_cumulative'),
                                        dmc.Checkbox(label='Total Tokens Cumulative', value='custom.total_tokens_cumulative'),
                                        dmc.Checkbox(label='Time to First Token', value='custom.time_to_first_token'),
                                    ],
                                    style={'background': '#f8fafc', 'padding': '12px', 'borderRadius': '8px', 'boxShadow': '0 1px 4px rgba(37,99,235,0.04)'},
                                ),
                            ]
                        ),
                        dmc.Button('Update Dashboard', id='update-button', n_clicks=0, style={
                            'height': '40px',
                            'backgroundColor': '#2563eb',
                            'color': 'white',
                            'border': 'none',
                            'padding': '0 20px',
                            'borderRadius': '8px',
                            'fontWeight': 600,
                            'fontSize': '1rem',
                            'boxShadow': '0 2px 8px rgba(37,99,235,0.12)',
                            'transition': 'background 0.2s',
                            'cursor': 'pointer',
                            'marginTop': '10px',
                        }),
                    ]
                ),

                # --- Main Dashboard Area ---
                html.Div(style={
                    'flex': 1,
                    'padding': '0 0 0 0',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'gap': '32px',
                }, children=[
                    # --- KPIs Section ---
                    html.Div([
                        html.H3("Key Performance Indicators", style={'color': '#2563eb', 'fontWeight': 600, 'marginBottom': '18px'}),
                        html.Div(id='kpi-container', style={
                            'display': 'grid',
                            'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))',
                            'gap': '24px',
                        })
                    ]),
                    # --- Graphs Section ---
                    html.Div([
                        html.H3("Visualizations", style={'color': '#2563eb', 'fontWeight': 600, 'marginBottom': '18px'}),
                        html.Div(id='graphs-container', style={
                            'display': 'grid',
                            'gridTemplateColumns': 'repeat(auto-fit, minmax(420px, 1fr))',
                            'gap': '32px',
                        })
                    ]),
                ]),
            ]),

            # --- Footer ---
            html.Footer("UbiOps Dashboard • Powered by Dash & Plotly • 2024", style={
                'textAlign': 'center',
                'padding': '18px 0 10px 0',
                'color': '#64748b',
                'fontSize': '1rem',
                'marginTop': '40px',
                'background': 'transparent',
                'letterSpacing': '1px',
            })
        ]
    )
)

# ==============================================================================
# 3. Dash Application Callbacks
# ==============================================================================

ALL_METRICS = [
    'deployments.requests',
    'deployments.failed_requests',
    'deployments.request_duration',
    'deployments.input_volume',
    'deployments.output_volume',
    'deployments.express_queue_time',
    'deployments.batch_queue_time',
    'deployments.credits',
    # 'deployments.instances',
    # 'deployments.express_queue_size',
    # 'deployments.batch_queue_size',
    'deployments.network_in',
    'deployments.network_out',
    # 'deployments.instance_start_time',
    # 'custom.token_count',
    'custom.total_tokens',
    'custom.prompt_tokens',
    'custom.completion_tokens',
    'custom.completion_tokens_cumulative',
    'custom.prompt_tokens_cumulative',
    'custom.total_tokens_cumulative',
    'custom.time_to_first_token',
]

@app.callback(
    Output("metric-selector", "value"),
    [Input("select-all-metrics", "n_clicks"),
     Input("deselect-all-metrics", "n_clicks")],
    State("metric-selector", "value"),
    prevent_initial_call=True
)
def select_deselect_all(select_all, deselect_all, current):
    triggered = ctx.triggered_id
    if triggered == "select-all-metrics":
        return ALL_METRICS
    elif triggered == "deselect-all-metrics":
        return []
    return current

@app.callback(
    [Output('kpi-container', 'children'),
     Output('graphs-container', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('project-selector', 'value'),
     State('start-datetime-picker', 'value'),
     State('end-datetime-picker', 'value'),
     State('aggregation-input', 'value'),
     State('metric-selector', 'value')]
)
def update_dashboard(n_clicks, project, start_datetime, end_datetime, aggregation_s, selected_metrics):
    if n_clicks == 0:
        return [html.P("Select your filters and click 'Update Dashboard' to begin.")], []

    # --- Parse inputs ---
    start_datetime_obj = parser.parse(start_datetime) if isinstance(start_datetime, str) else start_datetime
    end_datetime_obj = parser.parse(end_datetime) if isinstance(end_datetime, str) else end_datetime
    api = make_connection(project)
    labels_to_use = GEMMA_DEPLOYMENT_LABEL_POC if project == 'poc' else None

    # --- Metric Info Dictionary ---
    DEPLOYMENT_METRICS = {
        "deployments.credits": {"unit": "credits (float)", "description": "Usage of Credits"},
        # "deployments.instances": {"unit": "instances (float)", "description": "Average number of active deployment instances"},
        # "deployments.instance_start_time": {"unit": "seconds (float)", "description": "Average duration from instance creation to start time"},
        "deployments.input_volume": {"unit": "bytes (int)", "description": "Volume of incoming data in bytes"},
        "deployments.output_volume": {"unit": "bytes (int)", "description": "Volume of outgoing data in bytes"},
        "deployments.memory_utilization": {"unit": "bytes (int)", "description": "Peak memory used during a request"},
        "deployments.requests": {"unit": "requests (int)", "description": "Number of requests made to the object per minute"},
        "deployments.failed_requests": {"unit": "requests (int)", "description": "Number of failed requests made to the object per minute"},
        "deployments.request_duration": {"unit": "seconds (float)", "description": "Average time in seconds for a request to complete"},
        # "deployments.express_queue_size": {"unit": "items (int)", "description": "Average number of queued express requests"},
        "deployments.express_queue_time": {"unit": "items (int)", "description": "Average time in seconds for an express request to start processing"},
        # "deployments.batch_queue_size": {"unit": "items (int)", "description": "Average number of queued batch requests"},
        "deployments.batch_queue_time": {"unit": "items (int)", "description": "Average time in seconds for a batch request to start processing"},
        "deployments.network_in": {"unit": "bytes (int)", "description": "Inbound network traffic for a deployment version"},
        "deployments.network_out": {"unit": "bytes (int)", "description": "Outbound network traffic for a deployment version"},
        "custom.completion_tokens": {"unit": "tokens (int)", "description": "Total number of completion tokens"},
        "custom.prompt_tokens": {"unit": "tokens (int)", "description": "Total number of prompt tokens"},
        "custom.total_tokens": {"unit": "tokens (int)", "description": "Total number of tokens"},
        # "custom.token_count": {"unit": "tokens (int)", "description": "Total number of tokens"},
        "custom.completion_tokens_cumulative": {"unit": "tokens (int)", "description": "Cumulative completion tokens"},
        "custom.prompt_tokens_cumulative": {"unit": "tokens (int)", "description": "Cumulative prompt tokens"},
        "custom.total_tokens_cumulative": {"unit": "tokens (int)", "description": "Cumulative total tokens"},
        "custom.time_to_first_token": {"unit": "seconds (float)", "description": "Time to first token in seconds"},
    }

    # --- Fetch Data for Selected Metrics ---
    metric_dfs = {}
    for metric in selected_metrics:
        metric_dfs[metric] = get_time_series_metric(api, project, metric, start_datetime_obj, end_datetime_obj, aggregation_s, labels=labels_to_use)

    # --- Create KPI cards (optional, can be customized further) ---
    kpi_cards = []
    # Example: Show total requests if selected
    if 'deployments.requests' in metric_dfs and not metric_dfs['deployments.requests'].empty:
        # Calculate total requests by adjusting for aggregation window (rate * window_minutes)
        df = metric_dfs['deployments.requests']
        # aggregation_s is in seconds, convert to minutes
        window_minutes = aggregation_s / 60.0 if aggregation_s else 1.0
        total_requests = (df['value'] * window_minutes).sum()
        kpi_cards.append(html.Div(f"Total Requests: {total_requests:,.0f}", style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px', 'textAlign': 'center'}))
    if 'deployments.failed_requests' in metric_dfs and not metric_dfs['deployments.failed_requests'].empty:
        total_failed = metric_dfs['deployments.failed_requests']['value'].sum()
        kpi_cards.append(html.Div(f"Failed Requests: {total_failed:,.0f}", style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px', 'textAlign': 'center'}))
    if 'deployments.request_duration' in metric_dfs and not metric_dfs['deployments.request_duration'].empty:
        avg_duration = metric_dfs['deployments.request_duration']['value'].mean()
        kpi_cards.append(html.Div(f"Avg. Request Duration: {avg_duration:.2f}s", style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px', 'textAlign': 'center'}))
    if 'custom.time_to_first_token' in metric_dfs and not metric_dfs['custom.time_to_first_token'].empty:
        avg_ttf = metric_dfs['custom.time_to_first_token']['value'].mean()
        kpi_cards.append(html.Div(f"Avg. Time to First Token: {avg_ttf:.2f}s", style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px', 'textAlign': 'center'}))

    # --- Create Graphs for Selected Metrics ---
    graphs = []
    for metric in selected_metrics:
        df = metric_dfs[metric]
        if not df.empty and 'timestamp' in df.columns and 'value' in df.columns:
            metric_info = DEPLOYMENT_METRICS.get(metric, {})
            unit = metric_info.get('unit', '')
            description = metric_info.get('description', metric)
            title = f"{description}"
            yaxis_title = unit
            # Use bar for requests, failed_requests, else line
            if metric in ['deployments.requests', 'deployments.failed_requests']:
                fig = go.Figure(go.Bar(x=df['timestamp'], y=df['value'], name=title))
            else:
                fig = go.Figure(go.Scatter(x=df['timestamp'], y=df['value'], name=title, mode='lines'))
            # Add average line for request duration and time to first token
            if metric in ['deployments.request_duration', 'custom.time_to_first_token']:
                avg_value = df['value'].mean()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=[avg_value]*len(df),
                    mode='lines',
                    name='Average',
                    line=dict(color='red', width=2, dash='dot'),
                    showlegend=True
                ))
            fig.update_layout(title_text=title)
            fig.update_yaxes(title_text=yaxis_title)
            # Show legend inside the plot area, top right
            fig.update_layout(
                legend=dict(
                    x=0.99,
                    y=0.99,
                    xanchor='right',
                    yanchor='top',
                    bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1
                )
            )
            graphs.append(dcc.Graph(figure=fig))

    return kpi_cards, graphs

# ==============================================================================
# 4. Run the Application
# ==============================================================================

if __name__ == '__main__':
    app.run(debug=True)