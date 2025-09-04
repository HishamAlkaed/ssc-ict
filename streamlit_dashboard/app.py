import streamlit as st
import ubiops
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from dateutil import parser
import time
import numpy as np

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
        # if metric_name == "deployments.requests" and (end_date - start_date).total_seconds() <= 10*3600:
        #     print("hi", response)

        data_points = response.to_dict().get('data_points', [])
        if not data_points:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['timestamp', 'value'])
        df = pd.DataFrame([
            {"timestamp": pd.to_datetime(dp["end_date"]), "value": dp["value"]}
            for dp in data_points
        ])
        if metric_name in ["deployments.requests", "deployments.failed_requests"]:
            df['value'] = [round(x) for x in df['value'] * aggregation_s]

        return df
    except ubiops.exceptions.ApiException as e:
        st.error(f"API Error fetching '{metric_name}': {e}")
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
        st.error(f"API Error fetching deployment requests: {e}")
        return "Error"
    except Exception as ex:
        st.error(f"An error occurred in get_slow_requests_percentage: {ex}")
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

def get_time_window(period):
    now = datetime.datetime.now()
    if period == 'hour':
        return now - datetime.timedelta(hours=1), now
    elif period == 'day':
        return now - datetime.timedelta(days=1), now
    elif period == 'week':
        return now - datetime.timedelta(weeks=1), now
    elif period == 'month':
        return now - datetime.timedelta(days=30), now
    else:
        raise ValueError('Unknown period')

def fetch_ttf_fine_grained(api, project, labels_to_use, start, end):
    # Fetch custom.time_to_first_token in 1-day windows, aggregation=60
    dfs = []
    day = datetime.timedelta(days=1)
    current = start
    # print(start)
    while current < end:
        chunk_end = min(current + day, end)
        # print(chunk_end)
        df = get_time_series_metric(api, project, 'custom.time_to_first_token', current, chunk_end, 60, labels=labels_to_use)
        if not df.empty:
            dfs.append(df)
        current = chunk_end
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=['timestamp', 'value'])

def fetch_summary_stats(api, project, labels_to_use, period, ttf_threshold=3.0):
    start, end = get_time_window(period)
    # Choose aggregation period based on window length for requests/failed
    window_seconds = (end - start).total_seconds()
    if window_seconds <= 3700: # 1 hour
        agg = 60
    elif window_seconds <= 24*3600:  # 1 day
        agg = 300  # 5 min
    elif window_seconds <= 7*24*3600:  # 1 week
        agg = 3600  # 1 hour
    else:  # > 1 week (month)
        agg = 86400  # 1 day
    # print(f"start: {start}, \n end: {end}, \n agg: {agg}")
    df_requests = get_time_series_metric(api, project, 'deployments.requests', start, end, agg, labels=labels_to_use)
    # print(df_requests)
    # df_requests['value'] = [round(x) for x in df_requests['value'] * agg]

    df_failed = get_time_series_metric(api, project, 'deployments.failed_requests', start, end, agg, labels=labels_to_use)
    # df_failed['value'] = [round(x) for x in df_failed['value'] * agg]

    # For ttf, use fine-grained fetch
    df_ttf = fetch_ttf_fine_grained(api, project, labels_to_use, start, end)
    # print(df_ttf)

    total_requests = df_requests['value'].sum() if not df_requests.empty else 0
    total_failed = df_failed['value'].sum() if not df_failed.empty else 0
    # Happy/unhappy requests: count in ttf df
    happy = int((df_ttf['value'] < ttf_threshold).sum()) if not df_ttf.empty else 0
    unhappy = int((df_ttf['value'] >= ttf_threshold).sum()) if not df_ttf.empty else 0
    total_ttf = happy + unhappy
    happy_pct = (happy / total_ttf * 100) if total_ttf > 0 else 0
    unhappy_pct = (unhappy / total_ttf * 100) if total_ttf > 0 else 0
    # Average reaction time = average TTFT within selected period
    avg_ttf = float(df_ttf['value'].mean()) if not df_ttf.empty else 0.0
    return {
        'total_requests': total_requests,
        'total_failed': total_failed,
        'happy': happy,
        'unhappy': unhappy,
        'happy_pct': happy_pct,
        'unhappy_pct': unhappy_pct,
        'total_ttf': total_ttf,
        'avg_ttf': avg_ttf,
        'df_requests': df_requests,
        'df_failed': df_failed,
        'df_ttf': df_ttf,
    }

def get_peak_load_insight(df_requests, period):
    if df_requests.empty or 'timestamp' not in df_requests:
        return None
    # Group by hour of day
    df = df_requests.copy()
    df['hour'] = df['timestamp'].dt.hour
    hourly = df.groupby('hour')['value'].sum()
    if hourly.empty:
        return None
    peak_hour = hourly.idxmax()
    peak_count = hourly.max()
    # Find range if multiple hours have the same peak
    peak_hours = hourly[hourly == peak_count].index.tolist()
    if len(peak_hours) == 1:
        hour_str = f"{peak_hour:02d}:00 - {peak_hour+1:02d}:00"
    else:
        hour_str = f"{min(peak_hours):02d}:00 - {max(peak_hours)+1:02d}:00"
    period_str = {
        'hour': 'last hour',
        'day': 'last day',
        'week': 'last week',
        'month': 'last month',
    }[period]
    return f"Based on the usage of the {period_str}, the most requests and highest server load was between {hour_str}."

# ==============================================================================
# 2. Streamlit Application
# ==============================================================================

def create_detailed_plot(stats, period_label):
    """Create a detailed plot for the clicked period showing requests, failed, and unhappy requests"""
    period_map = {'Last Month': 'month', 'Last Week': 'week', 'Last Day': 'day', 'Last Hour': 'hour'}
    period = period_map[period_label]
    data = stats[period]
    
    # Get the dataframes
    df_requests = data['df_requests']
    df_failed = data.get('df_failed', pd.DataFrame())
    df_ttf = data.get('df_ttf', pd.DataFrame())
    
    if df_requests.empty:
        st.warning("No data available for this period")
        return
    
    # Create the main plot
    fig = go.Figure()
    
    # Add requests as bars
    fig.add_trace(go.Bar(
        x=df_requests['timestamp'],
        y=df_requests['value'],
        name='Requests',
        marker_color='#2563eb',
        opacity=0.8
    ))
    
    # Add failed requests as stacked bars
    if not df_failed.empty:
        fig.add_trace(go.Bar(
            x=df_failed['timestamp'],
            y=df_failed['value'],
            name='Failed Requests',
            marker_color='#dc2626',
            opacity=0.7
        ))
    
    # Add unhappy requests as vertical lines using add_shape
    if not df_ttf.empty:
        # Find timestamps where time to first token >= 3 seconds
        unhappy_times = df_ttf[df_ttf['value'] >= 3.0]['timestamp']
        if not unhappy_times.empty:
            # Get the y-axis range for the vertical lines
            y_max = df_requests['value'].max() if not df_requests.empty else 100
            
            # Create vertical lines for each unhappy request
            for i, time in enumerate(unhappy_times):
                fig.add_shape(
                    type="line",
                    x0=time,
                    x1=time,
                    y0=0,
                    y1=y_max,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash"
                    ),
                    name=f"Unhappy Request {i+1}"
                )
    
    # Update layout
    fig.update_layout(
        title=f"Detailed View: {period_label}",
        xaxis_title="Time",
        yaxis_title="Number of Requests",
        barmode='stack',  # Stack failed requests on top
        height=500,
        showlegend=True,
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )
    
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="UbiOps Server Usage Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid rgba(2, 6, 23, 0.06);
    }
    .metric-card h3 {
        margin: 0 0 0.25rem 0;
        font-weight: 600;
        color: #0f172a;
        font-size: 1rem;
    }
    .metric-card h2 {
        font-weight: 700;
        color: #2563eb;
    }
    .stSelectbox label {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # --- Always-visible summary cards and insights ---
    st.markdown("""
    <style>
    .summary-card {background: #ffffff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); padding: 1.1rem 1.3rem; margin-bottom: 0.5rem; text-align: center; border: 1px solid rgba(2, 6, 23, 0.06);} 
    .summary-title {font-size: 1.0rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem;}
    .summary-value {font-size: 1.4rem; font-weight: 700; margin: 0.15rem 0;}
    .summary-sub {font-size: 0.9rem; font-weight: 600; color: #0f172a; opacity: 0.8; margin-top: 0.35rem;}
    .summary-green {color: #16a34a;}
    .summary-red {color: #dc2626;}
    .summary-grey {color: #475569;}
    </style>
    """, unsafe_allow_html=True)

    # --- Summary Cards: load and display one-by-one ---
    periods = [('hour', 'Last Hour'), ('day', 'Last Day'), ('week', 'Last Week'), ('month', 'Last Month')]
    if 'summary_stats' not in st.session_state:
        st.session_state.summary_stats = {}
    if 'summary_stats_time' not in st.session_state:
        st.session_state.summary_stats_time = None
    if 'clicked_card' not in st.session_state:
        st.session_state.clicked_card = None

    update_summary = st.button("🔄 Update Performance Summary", key="update_summary_button")
    if update_summary:
        st.session_state.summary_stats = {}
        st.session_state.summary_stats_time = None

    card_cols = st.columns(len(periods))
    for idx, (period, label) in enumerate(periods):
        with card_cols[idx]:
            # Always show the button
            btn = st.button(f"📊 Show {label} Plot", key=f"card_{period}", use_container_width=True)
            
            # Load data if not already loaded
            if period not in st.session_state.summary_stats:
                with st.spinner(f"Loading {label}..."):
                    project = 'poc'
                    api = make_connection(project)
                    labels_to_use = GEMMA_DEPLOYMENT_LABEL_POC if project == 'poc' else None
                    s = fetch_summary_stats(api, project, labels_to_use, period)
                    st.session_state.summary_stats[period] = s
            
            # Show card content if data is available
            s = st.session_state.summary_stats.get(period)
            if s:
                st.markdown(
                    f"<div class='summary-card'>"
                    f"<div class='summary-title'>{label}</div>"
                    f"<div class='summary-value summary-grey'>Requests: {s['total_requests']:,}</div>"
                    f"<div class='summary-value summary-grey'>Failed: {s['total_failed']:,}</div>"
                    f"<div class='summary-value summary-green'>Happy: {s['happy']:,} ({s['happy_pct']:.0f}%)</div>"
                    f"<div class='summary-value summary-red'>Unhappy: {s['unhappy']:,} ({s['unhappy_pct']:.0f}%)</div>"
                    f"<div class='summary-sub'>Avg reaction time: {s['avg_ttf']:.2f}s</div>"
                    f"</div>", unsafe_allow_html=True
                )
            
            # Handle button click
            if btn:
                st.session_state.clicked_card = label

    if st.session_state.summary_stats:
        st.session_state.summary_stats_time = datetime.datetime.now()
        st.caption(f"Last updated: {st.session_state.summary_stats_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Show detailed plot if a card was clicked
    if st.session_state.clicked_card:
        with st.expander(f"📈 Detailed View: {st.session_state.clicked_card}", expanded=True):
            fig = create_detailed_plot(st.session_state.summary_stats, st.session_state.clicked_card)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            if st.button("❌ Close", key="close_plot"):
                st.session_state.clicked_card = None
                st.rerun()

    # --- Insights Section ---
    st.markdown("<h3 style='margin-top:1.5rem;'>🔎 Automated Insights</h3>", unsafe_allow_html=True)
    for period, label in periods:
        s = st.session_state.summary_stats.get(period)
        if s:
            insight = get_peak_load_insight(s['df_requests'], period)
            if insight:
                st.info(f"{label}: {insight}")

    # Sidebar for controls
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Project selector
        project = st.selectbox(
            "Select Project:",
            options=['poc', 'chat'],
            format_func=lambda x: 'poc (Gemma)' if x == 'poc' else 'chat (alles)',
            index=0
        )
        
        st.divider()
        
        # Date and time selectors
        st.subheader("📅 Time Range")
        start_date = st.date_input(
            "Start Date:",
            value=(datetime.datetime.now() - datetime.timedelta(days=1)).date()
        )
        start_time = st.time_input(
            "Start Time:",
            value=datetime.time(0, 0)
        )
        
        end_date = st.date_input(
            "End Date:",
            value=datetime.datetime.now().date()
        )
        end_time = st.time_input(
            "End Time:",
            value=datetime.datetime.now()
        )
        
        st.divider()
        
        # Aggregation selector
        st.subheader("⏱️ Aggregation")
        aggregation_options = {
            "1 minute": 60,
            "5 minutes": 300,
            "15 minutes": 900,
            "30 minutes": 1800,
            "1 hour": 3600,
            "6 hours": 21600,
            "12 hours": 43200,
            "1 day": 86400
        }
        aggregation_label = st.selectbox(
            "Aggregation Period:",
            options=list(aggregation_options.keys()),
            index=4  # Default to 1 hour
        )
        aggregation_s = aggregation_options[aggregation_label]
        
        st.divider()
        
        # Metric selector
        st.subheader("📈 Metrics to Display")
        
        # Metric categories
        deployment_metrics = {
            'Requests': 'deployments.requests',
            'Failed Requests': 'deployments.failed_requests',
            'Request Duration': 'deployments.request_duration',
            'Input Volume': 'deployments.input_volume',
            'Output Volume': 'deployments.output_volume',
            'Express Queue Time': 'deployments.express_queue_time',
            'Batch Queue Time': 'deployments.batch_queue_time',
            'Credits': 'deployments.credits',
            'Network In': 'deployments.network_in',
            'Network Out': 'deployments.network_out'
        }
        
        custom_metrics = {
            'Total Tokens': 'custom.total_tokens',
            'Prompt Tokens': 'custom.prompt_tokens',
            'Completion Tokens': 'custom.completion_tokens',
            'Completion Tokens Cumulative': 'custom.completion_tokens_cumulative',
            'Prompt Tokens Cumulative': 'custom.prompt_tokens_cumulative',
            'Total Tokens Cumulative': 'custom.total_tokens_cumulative',
            'Time to First Token': 'custom.time_to_first_token'
        }
        
        st.write("**Deployment Metrics:**")
        selected_deployment_metrics = st.multiselect(
            "Select deployment metrics:",
            options=list(deployment_metrics.keys()),
            default=['Requests'],
            key="deployment_metrics"
        )
        
        st.write("**Custom Metrics:**")
        selected_custom_metrics = st.multiselect(
            "Select custom metrics:",
            options=list(custom_metrics.keys()),
            default=[],
            key="custom_metrics"
        )
        
        # Combine selected metrics
        selected_metrics = [deployment_metrics[m] for m in selected_deployment_metrics] + \
                          [custom_metrics[m] for m in selected_custom_metrics]
        
        st.divider()
        
        # Update button
        if st.button("🔄 Update Dashboard", type="primary", use_container_width=True):
            st.session_state.update_dashboard = True
    
    # --- KPI Cards: load and display one-by-one ---
    if 'update_dashboard' not in st.session_state:
        st.session_state.update_dashboard = False
    if st.session_state.update_dashboard:
        with st.spinner("Fetching data from UbiOps API..."):
            # Parse datetime objects
            start_datetime = datetime.datetime.combine(start_date, start_time)
            end_datetime = datetime.datetime.combine(end_date, end_time)
            api = make_connection(project)
            labels_to_use = GEMMA_DEPLOYMENT_LABEL_POC if project == 'poc' else None
            DEPLOYMENT_METRICS = {
                "deployments.credits": {"unit": "credits (float)", "description": "Usage of Credits"},
                "deployments.input_volume": {"unit": "bytes (int)", "description": "Volume of incoming data in bytes"},
                "deployments.output_volume": {"unit": "bytes (int)", "description": "Volume of outgoing data in bytes"},
                "deployments.memory_utilization": {"unit": "bytes (int)", "description": "Peak memory used during a request"},
                "deployments.requests": {"unit": "requests (int)", "description": "Number of requests made to the object"},
                "deployments.failed_requests": {"unit": "requests (int)", "description": "Number of failed requests made to the object"},
                "deployments.request_duration": {"unit": "seconds (float)", "description": "Average time in seconds for a request to complete"},
                "deployments.express_queue_time": {"unit": "items (int)", "description": "Average time in seconds for an express request to start processing"},
                "deployments.batch_queue_time": {"unit": "items (int)", "description": "Average time in seconds for a batch request to start processing"},
                "deployments.network_in": {"unit": "bytes (int)", "description": "Inbound network traffic for a deployment version"},
                "deployments.network_out": {"unit": "bytes (int)", "description": "Outbound network traffic for a deployment version"},
                "custom.completion_tokens": {"unit": "tokens (int)", "description": "Total number of completion tokens"},
                "custom.prompt_tokens": {"unit": "tokens (int)", "description": "Total number of prompt tokens"},
                "custom.total_tokens": {"unit": "tokens (int)", "description": "Total number of tokens"},
                "custom.completion_tokens_cumulative": {"unit": "tokens (int)", "description": "Cumulative completion tokens"},
                "custom.prompt_tokens_cumulative": {"unit": "tokens (int)", "description": "Cumulative prompt tokens"},
                "custom.total_tokens_cumulative": {"unit": "tokens (int)", "description": "Cumulative total tokens"},
                "custom.time_to_first_token": {"unit": "seconds (float)", "description": "Time to first token in seconds"},
            }
            metric_keys = [
                'deployments.requests',
                'deployments.failed_requests',
                'deployments.request_duration',
                'custom.time_to_first_token',
            ]
            kpi_cols = st.columns(4)
            for i, metric in enumerate(metric_keys):
                kpi_placeholder = kpi_cols[i % 4].empty()
                with kpi_placeholder:
                    with st.spinner(f"Loading {DEPLOYMENT_METRICS[metric]['description']}..."):
                        df = get_time_series_metric(api, project, metric, start_datetime, end_datetime, aggregation_s, labels=labels_to_use)
                        value = None
                        icon = None
                        if metric == 'deployments.requests' and not df.empty:
                            value = f"{df['value'].sum():,.0f}"
                            icon = "📈"
                            title = "Total Requests"
                        elif metric == 'deployments.failed_requests' and not df.empty:
                            value = f"{df['value'].sum():,.0f}"
                            icon = "❌"
                            title = "Failed Requests"
                        elif metric == 'deployments.request_duration' and not df.empty:
                            value = f"{df['value'].mean():.2f}s"
                            icon = "⏱️"
                            title = "Avg. Request Duration"
                        elif metric == 'custom.time_to_first_token' and not df.empty:
                            value = f"{df['value'].mean():.2f}s"
                            icon = "⚡"
                            title = "Avg. Time to First Token"
                        if value:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{icon} {title}</h3>
                                <h2 style="color: #2563eb; margin: 0;">{value}</h2>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        # Show initial message
        st.info("👈 Use the sidebar controls to configure your dashboard and click 'Update Dashboard' to begin.")
        
        # Show sample data structure
        with st.expander("📋 Available Metrics"):
            st.write("""
            **Deployment Metrics:**
            - Requests: Number of requests
            - Failed Requests: Number of failed requests
            - Request Duration: Average request completion time
            - Input/Output Volume: Data transfer volumes
            - Queue Times: Express and batch queue processing times
            - Credits: Usage of credits
            - Network: Inbound and outbound traffic
            
            **Custom Metrics:**
            - Token counts (prompt, completion, total)
            - Cumulative token usage
            - Time to first token
            """)

def display_metrics_charts(metric_dfs, metrics_to_display, metric_info):
    """Display charts for the selected metrics"""
    if not metrics_to_display:
        st.info("No metrics selected for display.")
        return
    
    # Create columns for charts (2 per row)
    cols = st.columns(2)
    
    for i, metric in enumerate(metrics_to_display):
        df = metric_dfs.get(metric)
        
        if df is not None and not df.empty and 'timestamp' in df.columns and 'value' in df.columns:
            metric_info_dict = metric_info.get(metric, {})
            unit = metric_info_dict.get('unit', '')
            description = metric_info_dict.get('description', metric)
            
            with cols[i % 2]:
                st.subheader(f"📊 {description}")
                
                # Create the chart
                if metric in ['deployments.requests', 'deployments.failed_requests']:
                    # Bar chart for request counts
                    fig = px.bar(
                        df, 
                        x='timestamp', 
                        y='value',
                        title=description,
                        labels={'value': unit, 'timestamp': 'Time'}
                    )
                else:
                    # Line chart for other metrics
                    fig = px.line(
                        df, 
                        x='timestamp', 
                        y='value',
                        title=description,
                        labels={'value': unit, 'timestamp': 'Time'}
                    )
                
                # Add average line for duration metrics
                if metric in ['deployments.request_duration', 'custom.time_to_first_token']:
                    avg_value = df['value'].mean()
                    fig.add_hline(
                        y=avg_value, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Average: {avg_value:.2f}"
                    )
                
                # Update layout
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{df['value'].mean():.2f}")
                with col2:
                    st.metric("Max", f"{df['value'].max():.2f}")
                with col3:
                    st.metric("Min", f"{df['value'].min():.2f}")
        else:
            with cols[i % 2]:
                st.warning(f"No data available for {metric}")

if __name__ == "__main__":
    main() 