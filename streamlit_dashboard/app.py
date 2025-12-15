import streamlit as st
import ubiops
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import pytz
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 1. UbiOps API Configuration & Helper Functions
# ==============================================================================

try:
    dashboard_config = importlib.import_module("config")
except ModuleNotFoundError:
    dashboard_config = importlib.import_module("config_example")


def _get_config_value(name, default):
    return getattr(dashboard_config, name, default)


API_TOKENS = _get_config_value("API_TOKENS", {})
UBIOPS_HOST = _get_config_value("UBIOPS_HOST", "https://api.demo.vlam.ai/v2.1")
DEFAULT_TIME_RANGE_DAYS = max(1, int(_get_config_value("DEFAULT_TIME_RANGE_DAYS", 1)))
DEFAULT_AGGREGATION_SECONDS = int(_get_config_value("DEFAULT_AGGREGATION_SECONDS", 3600))
CHART_HEIGHT = int(_get_config_value("CHART_HEIGHT", 420))
CHART_MARGIN = _get_config_value("CHART_MARGIN", dict(l=20, r=20, t=40, b=20))
SLOW_REQUEST_THRESHOLD_SECONDS = float(_get_config_value("SLOW_REQUEST_THRESHOLD_SECONDS", 3.0))


def get_chart_margin():
    return dict(CHART_MARGIN) if isinstance(CHART_MARGIN, dict) else CHART_MARGIN


# Netherlands timezone configuration
NETHERLANDS_TZ = pytz.timezone('Europe/Amsterdam')


@st.cache_resource(show_spinner=False)
def make_connection(project_name):
    """Establish and cache a CoreApi client for the given project."""
    configuration = ubiops.Configuration()
    token = API_TOKENS.get(project_name)
    if not token:
        st.error(f"Missing API token for project '{project_name}'. Please update config.py.")
        st.stop()
    configuration.api_key['Authorization'] = token
    configuration.host = UBIOPS_HOST
    api_client = ubiops.ApiClient(configuration)
    return ubiops.CoreApi(api_client)

@st.cache_data(ttl=120, show_spinner=False)
def list_deployments_cached(project_name):
    """Return a cached list of deployment names in the project."""
    try:
        api = make_connection(project_name)
        deployments = api.deployments_list(project_name=project_name)
        names = []
        for d in deployments:
            if isinstance(d, dict):
                name = d.get('name')
            else:
                name = getattr(d, 'name', None)
            if name:
                names.append(name)
        return sorted(list(set(names)))
    except Exception as e:
        st.error(f"Failed to list deployments: {e}")
        return []

@st.cache_data(ttl=120, show_spinner=False)
def get_labels_for_deployments_cached(project_name, deployment_names):
    """Build a cached list of labels for all versions of the selected deployments."""
    api = make_connection(project_name)
    labels = []
    for dep_name in deployment_names:
        try:
            versions = api.deployment_versions_list(project_name=project_name, deployment_name=dep_name)
            for v in versions:
                if isinstance(v, dict):
                    vid = v.get('id') or v.get('version_id')
                else:
                    vid = getattr(v, 'id', None) or getattr(v, 'version_id', None)
                if vid:
                    labels.append(f"deployment_version_id:{vid}")
        except Exception:
            continue
    return labels

@st.cache_data(ttl=60, show_spinner=False)
def cached_time_series(project_name, metric_name, start_iso, end_iso, aggregation_s, labels_param):
    api = make_connection(project_name)
    resp = api.time_series_data_list(
        project_name=project_name,
        metric=metric_name,
        start_date=start_iso,
        end_date=end_iso,
        aggregation_period=aggregation_s,
        labels=labels_param
    )
    data_points = resp.to_dict().get('data_points', [])
    if not data_points:
        return pd.DataFrame(columns=['timestamp', 'value'])
    local_df = pd.DataFrame([
        {"timestamp": pd.to_datetime(dp["end_date"]), "value": dp["value"]}
        for dp in data_points
    ])
    return local_df

def _time_series_raw(api, project_name, metric_name, start_date, end_date, aggregation_s, labels=None):
    resp = api.time_series_data_list(
        project_name=project_name,
        metric=metric_name,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        aggregation_period=aggregation_s,
        labels=labels
    )
    data_points = resp.to_dict().get('data_points', [])
    if not data_points:
        return pd.DataFrame(columns=['timestamp', 'value'])
    return pd.DataFrame([
        {"timestamp": pd.to_datetime(dp["end_date"]), "value": dp["value"]}
        for dp in data_points
    ])

def get_time_series_metric(api, project_name, metric_name, start_date, end_date, aggregation_s, labels=None, use_streamlit_cache=True):
    """Fetches aggregated time-series data from UbiOps.
    If labels is a list, fetch per label and aggregate across them (sum for counts, mean for durations).
    """
    def fetch_single(labels_param):
        if use_streamlit_cache:
            local_df = cached_time_series(
                project_name,
                metric_name,
                start_date.isoformat(),
                end_date.isoformat(),
                aggregation_s,
                labels_param
            )
        else:
            local_df = _time_series_raw(
                api,
                project_name,
                metric_name,
                start_date,
                end_date,
                aggregation_s,
                labels_param
            )
        try:
            if isinstance(local_df["timestamp"].dtype, pd.DatetimeTZDtype):
                local_df["timestamp"] = local_df["timestamp"].dt.tz_convert(NETHERLANDS_TZ)
            else:
                local_df["timestamp"] = pd.to_datetime(local_df["timestamp"], utc=True).dt.tz_convert(NETHERLANDS_TZ)
        except Exception:
            pass
        if metric_name in ["deployments.requests", "deployments.failed_requests"]:
            local_df['value'] = [round(x) for x in local_df['value'] * aggregation_s]
        if metric_name == "custom.time_to_first_token_larger_3s":
            local_df['value'] = [round(x) for x in local_df['value'] * aggregation_s]
        return local_df

    try:
        # Aggregate across multiple labels if provided
        if isinstance(labels, list):
            dfs = []
            for lab in labels:
                df_lab = fetch_single(lab)
                if not df_lab.empty:
                    dfs.append(df_lab.rename(columns={'value': f"value_{len(dfs)}"}))
            if not dfs:
                return pd.DataFrame(columns=['timestamp', 'value'])
            merged = dfs[0]
            for other in dfs[1:]:
                merged = pd.merge(merged, other, on='timestamp', how='outer')
            # Fill missing with 0 for sums; for mean we will ignore NaNs
            value_cols = [c for c in merged.columns if c.startswith('value_')]
            if metric_name in ["deployments.request_duration", "custom.time_to_first_token"]:
                merged['value'] = merged[value_cols].mean(axis=1, skipna=True)
            else:
                merged[value_cols] = merged[value_cols].fillna(0)
                merged['value'] = merged[value_cols].sum(axis=1)
            return merged[['timestamp', 'value']].sort_values('timestamp').reset_index(drop=True)
        else:
            return fetch_single(labels)
    except ubiops.exceptions.ApiException as e:
        st.error(f"API Error fetching '{metric_name}': {e}")
        return pd.DataFrame(columns=['timestamp', 'value'])

 

def get_time_window(period):
    # Use Netherlands timezone and convert to UTC for API queries
    now_netherlands = datetime.datetime.now(NETHERLANDS_TZ)
    if period == 'hour':
        start_netherlands, end_netherlands = now_netherlands - datetime.timedelta(hours=1), now_netherlands
    elif period == 'day':
        start_netherlands, end_netherlands = now_netherlands - datetime.timedelta(days=1), now_netherlands
    elif period == 'week':
        start_netherlands, end_netherlands = now_netherlands - datetime.timedelta(weeks=1), now_netherlands
    elif period == 'month':
        start_netherlands, end_netherlands = now_netherlands - datetime.timedelta(days=30), now_netherlands
    else:
        raise ValueError('Unknown period')
    return start_netherlands.astimezone(datetime.timezone.utc), end_netherlands.astimezone(datetime.timezone.utc)

def fetch_metric_fine_grained(api, project, labels_to_use, start, end, metric_name):
    # Fetch a custom metric in 1-day windows, aggregation=60, concurrently
    day = datetime.timedelta(days=1)
    # Build chunk boundaries first
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + day, end)
        chunks.append((current, chunk_end))
        current = chunk_end

    if not chunks:
        return pd.DataFrame(columns=['timestamp', 'value'])

    # Submit all chunk requests in parallel
    dfs = []
    with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as executor:
        future_map = {
            executor.submit(
                get_time_series_metric,
                api,
                project,
                metric_name,
                chunk_start,
                chunk_end,
                60,
                labels_to_use,
                False
            ): (chunk_start, chunk_end)
            for (chunk_start, chunk_end) in chunks
        }
        for fut in as_completed(future_map):
            try:
                df = fut.result()
            except Exception:
                df = pd.DataFrame(columns=['timestamp', 'value'])
            if df is not None and not df.empty:
                dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=['timestamp', 'value'])
    # Concatenate and sort for stable output
    out = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in out.columns:
        out = out.sort_values('timestamp').reset_index(drop=True)
    return out

def fetch_summary_stats(api, project, labels_to_use, period, ttf_threshold=SLOW_REQUEST_THRESHOLD_SECONDS, use_streamlit_cache=True):
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
    df_requests = get_time_series_metric(api, project, 'deployments.requests', start, end, agg, labels=labels_to_use, use_streamlit_cache=use_streamlit_cache)
    # print(df_requests)
    # df_requests['value'] = [round(x) for x in df_requests['value'] * agg]

    df_failed = get_time_series_metric(api, project, 'deployments.failed_requests', start, end, agg, labels=labels_to_use, use_streamlit_cache=use_streamlit_cache)
    # df_failed['value'] = [round(x) for x in df_failed['value'] * agg]

    # For ttft (average reaction time), use fine-grained fetch
    df_ttf = fetch_metric_fine_grained(api, project, labels_to_use, start, end, 'custom.time_to_first_token')
    # For unhappy flags (0/1), use server-side metric
    df_unhappy_flags = fetch_metric_fine_grained(api, project, labels_to_use, start, end, 'custom.time_to_first_token_larger_3s')

    total_requests = df_requests['value'].sum() if not df_requests.empty else 0
    total_failed = df_failed['value'].sum() if not df_failed.empty else 0
    # Happy/unhappy requests: use server-side flags metric
    unhappy = int(df_unhappy_flags['value'].sum()) if not df_unhappy_flags.empty else 0
    happy = max(int(total_requests - unhappy), 0)
    total_ttf = happy + unhappy
    happy_pct = (happy / total_ttf * 100) if total_ttf > 0 else 0
    unhappy_pct = (unhappy / total_ttf * 100) if total_ttf > 0 else 0
    # Average reaction time = average TTFT within selected period
    avg_ttf = float(df_ttf['value'].mean()) if not df_ttf.empty else 0.0
    # Compute token totals for unhappy time buckets (where TTFT >= threshold)
    unhappy_token_totals = { 'total': 0, 'prompt': 0, 'completion': 0 }
    unhappy_token_avgs = { 'total': 0.0, 'prompt': 0.0, 'completion': 0.0 }
    avg_unhappy_ttf = 0.0
    # Determine unhappy timestamps from server-side flags
    unhappy_times = set(df_unhappy_flags[df_unhappy_flags['value'] >= 1]['timestamp']) if not df_unhappy_flags.empty else set()
    if unhappy_times:
        # average TTFT among unhappy using ttft values at unhappy timestamps
        if not df_ttf.empty:
            df_ttf_unhappy = df_ttf[df_ttf['timestamp'].isin(unhappy_times)]
            if not df_ttf_unhappy.empty:
                avg_unhappy_ttf = float(df_ttf_unhappy['value'].mean())
        # Fetch token metrics ONLY when there are unhappy requests
        df_total_tokens = fetch_metric_fine_grained(api, project, labels_to_use, start, end, 'custom.total_tokens')
        df_prompt_tokens = fetch_metric_fine_grained(api, project, labels_to_use, start, end, 'custom.prompt_tokens')
        df_completion_tokens = fetch_metric_fine_grained(api, project, labels_to_use, start, end, 'custom.completion_tokens')
        # Sum tokens at unhappy timestamps
        if not df_total_tokens.empty:
            unhappy_token_totals['total'] = int(df_total_tokens[df_total_tokens['timestamp'].isin(unhappy_times)]['value'].sum())
        if not df_prompt_tokens.empty:
            unhappy_token_totals['prompt'] = int(df_prompt_tokens[df_prompt_tokens['timestamp'].isin(unhappy_times)]['value'].sum())
        if not df_completion_tokens.empty:
            unhappy_token_totals['completion'] = int(df_completion_tokens[df_completion_tokens['timestamp'].isin(unhappy_times)]['value'].sum())
        # Averages per unhappy request
        if unhappy > 0:
            unhappy_token_avgs['total'] = unhappy_token_totals['total'] / unhappy
            unhappy_token_avgs['prompt'] = unhappy_token_totals['prompt'] / unhappy
            unhappy_token_avgs['completion'] = unhappy_token_totals['completion'] / unhappy

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
        'df_unhappy': df_unhappy_flags,
        'unhappy_total_tokens': unhappy_token_totals['total'],
        'unhappy_prompt_tokens': unhappy_token_totals['prompt'],
        'unhappy_completion_tokens': unhappy_token_totals['completion'],
        'unhappy_avg_total_tokens': unhappy_token_avgs['total'],
        'unhappy_avg_prompt_tokens': unhappy_token_avgs['prompt'],
        'unhappy_avg_completion_tokens': unhappy_token_avgs['completion'],
        'avg_unhappy_ttf': avg_unhappy_ttf,
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
    df_unhappy = data.get('df_unhappy', pd.DataFrame())
    
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
    
    # Add unhappy requests as vertical lines using server-side flags
    if df_unhappy is not None and not df_unhappy.empty:
        unhappy_times = df_unhappy[df_unhappy['value'] >= 1]['timestamp']
        if not unhappy_times.empty:
            y_max = df_requests['value'].max() if not df_requests.empty else 100
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
        height=CHART_HEIGHT + 100,
        showlegend=True,
        margin=get_chart_margin(),
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
    
    # Initialize persistent defaults in session state (only once per session)
    if 'start_date' not in st.session_state:
        st.session_state.start_date = (datetime.datetime.now(NETHERLANDS_TZ) - datetime.timedelta(days=DEFAULT_TIME_RANGE_DAYS)).date()
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.time(0, 0)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.datetime.now(NETHERLANDS_TZ).date()
    if 'end_time' not in st.session_state:
        st.session_state.end_time = datetime.datetime.now(NETHERLANDS_TZ).time().replace(microsecond=0)
    if 'agg_days' not in st.session_state:
        st.session_state.agg_days = DEFAULT_TIME_RANGE_DAYS

    # Sidebar for controls (placed first so selections affect all sections)
    with st.sidebar:
        st.header("📊 Dashboard Controls")

        # Deployment selector
        project = 'poc'
        api = make_connection(project)
        all_deployments = list_deployments_cached(project)
        # Initialize default selection on first load
        if 'selected_deployments' not in st.session_state:
            if 'vlam-chat-mistral-medium' in all_deployments:
                st.session_state.selected_deployments = ['vlam-chat-mistral-medium']
            else:
                st.session_state.selected_deployments = all_deployments[:1] if all_deployments else []

        # Dropdown-style popover containing searchable checkboxes and select-all controls
        # with st.popover("Select deployments", width='stretch'):
        with st.popover("Select deployments"):
            search_term = st.text_input("Search deployments", value=st.session_state.get('deployments_search', ''), key="deployments_search")
            filtered_deployments = [d for d in all_deployments if (search_term.lower() in d.lower())]

            cols_sa = st.columns(2)
            with cols_sa[0]:
                if st.button("Select all (filtered)"):
                    st.session_state.selected_deployments = sorted(list(set(st.session_state.selected_deployments).union(filtered_deployments)))
                    # Keep checkbox widget states in sync so they don't override on rerun
                    for dep in filtered_deployments:
                        st.session_state[f"depchk_{dep}"] = True
            with cols_sa[1]:
                if st.button("Clear all (filtered)"):
                    st.session_state.selected_deployments = sorted(list(set(st.session_state.selected_deployments) - set(filtered_deployments)))
                    # Keep checkbox widget states in sync so they don't override on rerun
                    for dep in filtered_deployments:
                        st.session_state[f"depchk_{dep}"] = False

            current_selection = set(st.session_state.selected_deployments)
            updated_selection_filtered = set()
            for dep in filtered_deployments:
                checked = dep in current_selection
                # Prefer explicit widget state if present; otherwise use derived selection
                default_val = st.session_state.get(f"depchk_{dep}", checked)
                is_checked = st.checkbox(dep, value=default_val, key=f"depchk_{dep}")
                if is_checked:
                    updated_selection_filtered.add(dep)

            preserved = current_selection - set(filtered_deployments)
            st.session_state.selected_deployments = sorted(list(preserved.union(updated_selection_filtered)))

        selected_deployments = st.session_state.selected_deployments
        st.caption(f"Selected {len(selected_deployments)} of {len(all_deployments)} deployments")

        # Aggregated comparison view toggle
        aggregated_view = st.checkbox("Aggregated view (compare deployments)", value=False, key="aggregated_view")
        if aggregated_view:
            st.number_input(
                "Days to include (for 'Last N Days' section)",
                min_value=1,
                max_value=60,
                value=st.session_state.get('agg_days', DEFAULT_TIME_RANGE_DAYS),
                step=1,
                key="agg_days"
            )

        st.divider()

        # Date and time selectors
        st.subheader("📅 Time Range")
        start_date = st.date_input(
            "Start Date:",
            value=st.session_state.start_date,
            key="start_date"
        )
        start_time = st.time_input(
            "Start Time:",
            value=st.session_state.start_time,
            key="start_time"
        )
        
        end_date = st.date_input(
            "End Date:",
            value=st.session_state.end_date,
            key="end_date"
        )
        # Use session state's time value so it doesn't reset on rerun
        end_time = st.time_input(
            "End Time:",
            value=st.session_state.end_time,
            key="end_time"
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
        aggregation_labels = list(aggregation_options.keys())
        default_agg_index = next(
            (i for i, label in enumerate(aggregation_labels) if aggregation_options[label] == DEFAULT_AGGREGATION_SECONDS),
            aggregation_labels.index("1 hour") if "1 hour" in aggregation_options else 0
        )
        aggregation_label = st.selectbox(
            "Aggregation Period:",
            options=aggregation_labels,
            index=default_agg_index
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
        # if st.button("🔄 Update Dashboard", type="primary", width='stretch'):
        if st.button("🔄 Update Dashboard", type="primary"):
            st.session_state.update_dashboard = True

    # Determine labels for summary based on selected deployments
    api_for_summary = make_connection('poc')
    labels_to_use_summary = get_labels_for_deployments_cached('poc', st.session_state.get('selected_deployments', []))

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

    if not st.session_state.get('aggregated_view', False):
        # Render placeholders for each period
        card_cols = st.columns(len(periods))
        card_placeholders = {}
        for idx, (period, label) in enumerate(periods):
            with card_cols[idx]:
                # button per card
                # btn = st.button(f"📊 Show {label} Plot", key=f"card_{period}", width='stretch')
                btn = st.button(f"📊 Show {label} Plot", key=f"card_{period}")
                if btn:
                    st.session_state.clicked_card = label
                card_placeholders[period] = st.empty()

        # Kick off all missing periods in parallel and update as they complete
        missing_periods = [p for p, _ in periods if p not in st.session_state.summary_stats]
        if missing_periods:
            api = make_connection(project)
            with ThreadPoolExecutor(max_workers=len(missing_periods)) as executor:
                future_map = {executor.submit(fetch_summary_stats, api, project, labels_to_use_summary, p, SLOW_REQUEST_THRESHOLD_SECONDS, False): p for p in missing_periods}
                for fut in as_completed(future_map):
                    p = future_map[fut]
                    try:
                        s = fut.result()
                    except Exception:
                        s = None
                    st.session_state.summary_stats[p] = s
                    # Update the matching card immediately
                    if s is not None:
                        label = dict(periods)[p]
                        card_placeholders[p].markdown(
                            f"<div class='summary-card'>"
                            f"<div class='summary-title'>{label} (Deployments: {', '.join(st.session_state.get('selected_deployments', []) or ['None'])})</div>"
                            f"<div class='summary-value summary-grey'>Requests: {s['total_requests']:,}</div>"
                            f"<div class='summary-value summary-grey'>Failed: {s['total_failed']:,}</div>"
                            f"<div class='summary-value summary-green'>Happy: {s['happy']:,} ({s['happy_pct']:.0f}%)</div>"
                            f"<div class='summary-value summary-red'>Unhappy: {s['unhappy']:,} ({s['unhappy_pct']:.0f}%)</div>"
                            + (f"<div class='summary-sub'>Avg Unhappy Tokens (total/prompt/completion): {s.get('unhappy_avg_total_tokens', 0):.1f} / {s.get('unhappy_avg_prompt_tokens', 0):.1f} / {s.get('unhappy_avg_completion_tokens', 0):.1f}</div>" if s.get('unhappy', 0) > 0 else "")
                            + (f"<div class='summary-sub'>Avg reaction time (unhappy only): {s.get('avg_unhappy_ttf', 0.0):.2f}s</div>" if s.get('unhappy', 0) > 0 else "")
                            + f"<div class='summary-sub'>Avg reaction time: {s['avg_ttf']:.2f}s</div>"
                            f"</div>", unsafe_allow_html=True
                        )
                        # Auto-open last hour plot when ready (only first time)
                        if p == 'hour' and not st.session_state.get('auto_opened_hour', False):
                            st.session_state.clicked_card = 'Last Hour'
                            st.session_state.auto_opened_hour = True

        # Fill any periods that were already cached on previous runs
        for p, label in periods:
            s = st.session_state.summary_stats.get(p)
            if s is not None and card_placeholders.get(p):
                card_placeholders[p].markdown(
                    f"<div class='summary-card'>"
                    f"<div class='summary-title'>{label} (Deployments: {', '.join(st.session_state.get('selected_deployments', []) or ['None'])})</div>"
                    f"<div class='summary-value summary-grey'>Requests: {s['total_requests']:,}</div>"
                    f"<div class='summary-value summary-grey'>Failed: {s['total_failed']:,}</div>"
                    f"<div class='summary-value summary-green'>Happy: {s['happy']:,} ({s['happy_pct']:.0f}%)</div>"
                    f"<div class='summary-value summary-red'>Unhappy: {s['unhappy']:,} ({s['unhappy_pct']:.0f}%)</div>"
                    + (f"<div class='summary-sub'>Avg Unhappy Tokens (total/prompt/completion): {s.get('unhappy_avg_total_tokens', 0):.1f} / {s.get('unhappy_avg_prompt_tokens', 0):.1f} / {s.get('unhappy_avg_completion_tokens', 0):.1f}</div>" if s.get('unhappy', 0) > 0 else "")
                    + (f"<div class='summary-sub'>Avg reaction time (unhappy only): {s.get('avg_unhappy_ttf', 0.0):.2f}s</div>" if s.get('unhappy', 0) > 0 else "")
                    + f"<div class='summary-sub'>Avg reaction time: {s['avg_ttf']:.2f}s</div>"
                    f"</div>", unsafe_allow_html=True
                )

    if not st.session_state.get('aggregated_view', False):
        if st.session_state.summary_stats:
            st.session_state.summary_stats_time = datetime.datetime.now(NETHERLANDS_TZ)
            st.caption(f"Last updated: {st.session_state.summary_stats_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if not st.session_state.get('aggregated_view', False) and st.session_state.clicked_card:
        with st.expander(f"📈 Detailed View: {st.session_state.clicked_card}", expanded=True):
            fig = create_detailed_plot(st.session_state.summary_stats, st.session_state.clicked_card)
            if fig:
                st.plotly_chart(fig, width='stretch', key=f"detail_plot_{st.session_state.clicked_card}")
            if st.button("❌ Close", key="close_plot"):
                st.session_state.clicked_card = None
                st.rerun()

    # --- Insights Section ---
    if not st.session_state.get('aggregated_view', False):
        st.markdown("<h3 style='margin-top:1.5rem;'>🔎 Automated Insights</h3>", unsafe_allow_html=True)
        for period, label in periods:
            s = st.session_state.summary_stats.get(period)
            if s:
                insight = get_peak_load_insight(s['df_requests'], period)
                if insight:
                    st.info(f"{label}: {insight}")

    # --- Aggregated comparison view ---
    if st.session_state.get('aggregated_view', False):
        api_cmp = make_connection('poc')
        selected = st.session_state.get('selected_deployments', [])
        if not selected:
            st.info("Select one or more deployments to compare.")
        else:
            st.markdown("<h3>🧮 Aggregated Comparison</h3>", unsafe_allow_html=True)

            # Helper to build per-deployment labels
            def labels_for_dep(dep_name):
                return get_labels_for_deployments_cached('poc', [dep_name])

            # Define two sections: Last Hour and Last N Days
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            start_hour = (now_utc - datetime.timedelta(hours=1))
            start_days = (now_utc - datetime.timedelta(days=st.session_state.get('agg_days', DEFAULT_TIME_RANGE_DAYS)))

            sections = [
                ("Last Hour", start_hour, now_utc),
                (f"Last {st.session_state.get('agg_days', 1)} Days", start_days, now_utc)
            ]

            for section_title, start_dt, end_dt in sections:
                with st.expander(section_title, expanded=True):
                    total_steps = max(len(selected) * 5, 1)
                    step_count = 0
                    status_line = st.empty()
                    progress = st.progress(0)

                    # Plot 1: Requests and Failed Requests
                    status_line.info("Fetching requests and failed requests in parallel...")
                    fig1 = go.Figure()
                    with ThreadPoolExecutor(max_workers=min(8, max(1, len(selected) * 2))) as executor:
                        futures_r = {executor.submit(get_time_series_metric, api_cmp, 'poc', 'deployments.requests', start_dt, end_dt, aggregation_s, labels_for_dep(dep), False): (dep, 'r') for dep in selected}
                        futures_f = {executor.submit(get_time_series_metric, api_cmp, 'poc', 'deployments.failed_requests', start_dt, end_dt, aggregation_s, labels_for_dep(dep), False): (dep, 'f') for dep in selected}
                        for fut in as_completed({**futures_r, **futures_f}):
                            dep, kind = ({**futures_r, **futures_f})[fut]
                            try:
                                df = fut.result()
                            except Exception:
                                df = None
                            step_count += 1; progress.progress(min(step_count / total_steps, 1.0))
                            if df is not None and not df.empty:
                                if kind == 'r':
                                    fig1.add_trace(go.Bar(x=df['timestamp'], y=df['value'], name=f"{dep} - Requests"))
                                else:
                                    fig1.add_trace(go.Bar(x=df['timestamp'], y=df['value'], name=f"{dep} - Failed"))
                    fig1.update_layout(title="Requests and Failed Requests", xaxis_title="Time", yaxis_title="Count", height=CHART_HEIGHT, showlegend=True, margin=get_chart_margin(), barmode='group')
                    st.plotly_chart(fig1, width='stretch', key=f"agg_{section_title}_fig1")

                    # Plot 2: Token counts (Prompt vs Completion)
                    status_line.info("Fetching token counts in parallel...")
                    fig2 = go.Figure()
                    with ThreadPoolExecutor(max_workers=min(8, max(1, len(selected) * 2))) as executor:
                        futures_p = {executor.submit(get_time_series_metric, api_cmp, 'poc', 'custom.prompt_tokens', start_dt, end_dt, aggregation_s, labels_for_dep(dep), False): (dep, 'p') for dep in selected}
                        futures_c = {executor.submit(get_time_series_metric, api_cmp, 'poc', 'custom.completion_tokens', start_dt, end_dt, aggregation_s, labels_for_dep(dep), False): (dep, 'c') for dep in selected}
                        for fut in as_completed({**futures_p, **futures_c}):
                            dep, kind = ({**futures_p, **futures_c})[fut]
                            try:
                                df = fut.result()
                            except Exception:
                                df = None
                            step_count += 1; progress.progress(min(step_count / total_steps, 1.0))
                            if df is not None and not df.empty:
                                if kind == 'p':
                                    fig2.add_trace(go.Bar(x=df['timestamp'], y=df['value'], name=f"{dep} - Prompt Tokens"))
                                else:
                                    fig2.add_trace(go.Bar(x=df['timestamp'], y=df['value'], name=f"{dep} - Completion Tokens"))
                    fig2.update_layout(title="Token Counts (Prompt & Completion)", xaxis_title="Time", yaxis_title="Tokens", height=CHART_HEIGHT, showlegend=True, margin=get_chart_margin(), barmode='group')
                    st.plotly_chart(fig2, width='stretch', key=f"agg_{section_title}_fig2")

                    # Plot 3: Reaction Time (TTFT)
                    status_line.info("Fetching reaction time in parallel...")
                    fig3 = go.Figure()
                    with ThreadPoolExecutor(max_workers=min(8, max(1, len(selected)))) as executor:
                        futures_t = {executor.submit(get_time_series_metric, api_cmp, 'poc', 'custom.time_to_first_token', start_dt, end_dt, aggregation_s, labels_for_dep(dep), False): dep for dep in selected}
                        for fut in as_completed(futures_t):
                            dep = futures_t[fut]
                            try:
                                df = fut.result()
                            except Exception:
                                df = None
                            step_count += 1; progress.progress(min(step_count / total_steps, 1.0))
                            if df is not None and not df.empty:
                                fig3.add_trace(go.Bar(x=df['timestamp'], y=df['value'], name=f"{dep} - TTFT"))
                    fig3.update_layout(title="Reaction Time (Time to First Token)", xaxis_title="Time", yaxis_title="Seconds", height=CHART_HEIGHT, showlegend=True, margin=get_chart_margin(), barmode='group')
                    st.plotly_chart(fig3, width='stretch', key=f"agg_{section_title}_fig3")

                    # Clear status and progress when done
                    status_line.empty()
                    progress.empty()
    
    # --- KPI Cards: load and display one-by-one ---
    if 'update_dashboard' not in st.session_state:
        st.session_state.update_dashboard = False
    if st.session_state.update_dashboard:
        with st.spinner("Fetching data from UbiOps API..."):
            # Parse datetime objects in Netherlands timezone and convert to UTC for API
            start_netherlands = NETHERLANDS_TZ.localize(datetime.datetime.combine(start_date, start_time))
            end_netherlands = NETHERLANDS_TZ.localize(datetime.datetime.combine(end_date, end_time))
            start_datetime = start_netherlands.astimezone(datetime.timezone.utc)
            end_datetime = end_netherlands.astimezone(datetime.timezone.utc)
            api = make_connection(project)
            labels_to_use = get_labels_for_deployments_cached(project, st.session_state.get('selected_deployments', []))
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
            # Fetch KPI metrics in parallel
            with ThreadPoolExecutor(max_workers=len(metric_keys)) as executor:
                future_map = {executor.submit(get_time_series_metric, api, project, metric, start_datetime, end_datetime, aggregation_s, labels_to_use, False): metric for metric in metric_keys}
                results = {}
                for fut in as_completed(future_map):
                    metric = future_map[fut]
                    try:
                        results[metric] = fut.result()
                    except Exception:
                        results[metric] = pd.DataFrame(columns=['timestamp','value'])

            for i, metric in enumerate(metric_keys):
                df = results.get(metric, pd.DataFrame(columns=['timestamp','value']))
                kpi_placeholder = kpi_cols[i % 4].empty()
                with kpi_placeholder:
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

            # Fetch and render selected metric charts
            st.divider()
            st.subheader("Selected Metrics")

            # Build a combined metric info dict (deployment + custom are already included)
            metric_info = DEPLOYMENT_METRICS

            # Collect dataframes for selected metrics
            metric_dfs = {}
            for metric in selected_metrics:
                try:
                    metric_dfs[metric] = get_time_series_metric(
                        api,
                        project,
                        metric,
                        start_datetime,
                        end_datetime,
                        aggregation_s,
                        labels=labels_to_use
                    )
                except Exception as ex:
                    st.warning(f"Failed to load {metric}: {ex}")

            # Render charts
            display_metrics_charts(metric_dfs, selected_metrics, metric_info)

            # Reset flag so we don't refetch until user clicks again
            st.session_state.update_dashboard = False
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
                    height=CHART_HEIGHT,
                    showlegend=False,
                    margin=get_chart_margin()
                )
                
                st.plotly_chart(fig, width='stretch', key=f"metric_plot_{metric}")
                
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