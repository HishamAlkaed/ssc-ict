# UbiOps API Configuration Example
# Copy this file to config.py and fill in your actual values

# API Tokens - Replace with your actual UbiOps API tokens
API_TOKENS = {
    "poc": "Token YOUR_POC_TOKEN_HERE",  # Replace with your POC token
    "chat": "Token YOUR_CHAT_TOKEN_HERE"  # Replace with your Chat token
}

# UbiOps API Host (optional, defaults to demo instance)
UBIOPS_HOST = "https://api.demo.vlam.ai/v2.1"

# Deployment Label for POC project (optional)
# This filters metrics for a specific deployment version
GEMMA_DEPLOYMENT_LABEL_POC = "deployment_version_id:YOUR_DEPLOYMENT_ID_HERE"

# Default time range (in days)
DEFAULT_TIME_RANGE_DAYS = 1

# Default aggregation period (in seconds)
DEFAULT_AGGREGATION_SECONDS = 3600  # 1 hour

# Chart configuration
CHART_HEIGHT = 400
CHART_MARGIN = dict(l=20, r=20, t=40, b=20)

# KPI threshold values
SLOW_REQUEST_THRESHOLD_SECONDS = 3.0 