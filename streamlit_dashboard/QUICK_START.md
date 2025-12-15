# 🚀 Quick Start Guide

Spin up the dashboard in minutes and start comparing deployments.

## 1. Install & Activate

```bash
cd streamlit_dashboard
python -m venv .venv && .venv/Scripts/activate  # optional
pip install -r requirements.txt
```

## 2. Supply Credentials

- Rename `config_example.py` to `config.py` (or export env vars) and add your UbiOps tokens.
- Minimum requirement inside `config.py`:

```python
API_TOKENS = {"poc": "Token YOUR_ACTUAL_POC_TOKEN"}
```

Add more project tokens or deployment labels as needed.

## 3. Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501`, pick deployments/time ranges in the sidebar, and hit **Update Dashboard**.

---

## 🎯 What You’ll See

1. **Sidebar controls** for project, deployments, time window, aggregation, and metrics.
2. **KPI cards** with action buttons that expand into detailed Plotly charts.
3. **Aggregated comparison view** with grouped bars for requests, failures, tokens, and TTFT.
4. **Metric explorer** that renders quick bar/line charts plus mean/max/min stats.

## 🔧 Customize

- Add/rename metrics in the sidebar dictionaries inside `app.py`.
- Adjust styling via the CSS block near the top of `app.py`.
- Point `make_connection` to another UbiOps instance if required.

## 🆘 Need Help?

- Read the compact [README](README.md) for setup notes and tips.
- Validate tokens and connectivity if charts stay empty.
- Reach out through your usual project channel.

---

That’s it — your dashboard should be live at `http://localhost:8501` 🎉