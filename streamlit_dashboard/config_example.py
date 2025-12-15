"""
Example configuration for the UbiOps Server Usage Dashboard.

Copy this file to config.py and replace the placeholder values with your own.
"""

# API tokens grouped by project. The dashboard currently references the "poc" project.
API_TOKENS = {
    "poc": "Token YOUR_POC_TOKEN_HERE",
    # "chat": "Token YOUR_CHAT_TOKEN_HERE",
}

# Optional: point to a different UbiOps instance / environment.
UBIOPS_HOST = "https://api.demo.vlam.ai/v2.1"

# Defaults for new sessions
DEFAULT_TIME_RANGE_DAYS = 1                # start_date defaults to now - N days
DEFAULT_AGGREGATION_SECONDS = 3600         # default aggregation option in the UI

# Plot styling
CHART_HEIGHT = 420
CHART_MARGIN = dict(l=20, r=20, t=40, b=20)

# Threshold to flag slow requests / unhappy sessions in insights
SLOW_REQUEST_THRESHOLD_SECONDS = 3.0