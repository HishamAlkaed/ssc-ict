# UbiOps Server Usage Dashboard

Compact Streamlit app to monitor UbiOps deployment health and token usage with minimal setup.

## Core Functionality

- **Aggregated comparison**: Expand “Last Hour” or “Last N Days” to compare requests, failed requests, token volumes, and time-to-first-token across deployments using grouped bar charts.
- **KPI drill-downs**: Hour/day/week/month cards open interactive Plotly line charts so you can inspect spikes and trends.
- **Metric explorer**: Pick any deployment/custom metric, set aggregation (1 min–1 day), and view quick charts with mean/max/min stats.
- **Automated insights**: Highlights peak-load windows plus TTFT anomalies for fast troubleshooting.

## Setup

```bash
cd streamlit_dashboard
python -m venv .venv && .venv/Scripts/activate  # optional
pip install -r requirements.txt
```

Configure credentials in `app.py` (prefer environment variables for production):

```python
API_TOKENS = {"poc": "Token YOUR_TOKEN"}
```

Adjust deployment label constants if you need version-specific filtering.

## Run

```bash
streamlit run app.py
```

Browse to `http://localhost:8501`, choose deployments/time ranges in the sidebar, and click **Update Dashboard**.

## Tips

- Keep tokens out of version control; mirror `config_example.py` if you need a local config.
- Shorter time windows or higher aggregation keeps charts responsive.
- `QUICK_START.md` includes screenshots and extended walkthroughs.

