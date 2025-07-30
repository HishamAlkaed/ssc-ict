#!/bin/bash

echo "🚀 UbiOps Server Usage Dashboard"
echo "================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found in current directory"
    echo "Please run this script from the streamlit_dashboard directory"
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing/updating dependencies..."
pip3 install -r requirements.txt

# Check configuration
if [ -f "config.py" ]; then
    echo "✅ Configuration file found"
else
    echo "⚠️  config.py not found - using default configuration"
    echo "   Copy config_example.py to config.py and update with your API tokens"
fi

echo
echo "🚀 Starting Streamlit application..."
echo "The dashboard will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo

# Start Streamlit
python3 -m streamlit run app.py

echo
echo "�� Dashboard stopped" 