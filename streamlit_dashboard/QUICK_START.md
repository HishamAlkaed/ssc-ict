# 🚀 Quick Start Guide

## Get the Dashboard Running in 3 Steps

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Configure API Tokens** (Optional)
Copy `config_example.py` to `config.py` and update with your UbiOps API tokens:
```python
API_TOKENS = {
    "poc": "Token YOUR_ACTUAL_POC_TOKEN",
    "chat": "Token YOUR_ACTUAL_CHAT_TOKEN"
}
```

### 3. **Run the Dashboard**

**Windows:**
```bash
run_dashboard.bat
```

**Mac/Linux:**
```bash
./run_dashboard.sh
```

**Or manually:**
```bash
streamlit run app.py
```

---

## 🎯 What You'll See

1. **Sidebar Controls**: Select project, time range, and metrics
2. **KPI Cards**: Key performance indicators at the top
3. **Interactive Charts**: Beautiful visualizations of your data
4. **Real-time Updates**: Click "Update Dashboard" to fetch fresh data

## 🔧 Customization

- **Add Metrics**: Edit the metric dictionaries in `app.py`
- **Change Styling**: Modify the CSS in the `st.markdown` section
- **Configure API**: Update the `make_connection` function for different UbiOps instances

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed instructions
- Review the [troubleshooting section](README.md#troubleshooting)
- Ensure your API tokens are correct and UbiOps service is accessible

---

**That's it!** Your dashboard should be running at `http://localhost:8501` 🎉 