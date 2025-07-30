@echo off
echo 🚀 UbiOps Server Usage Dashboard
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app.py" (
    echo ❌ app.py not found in current directory
    echo Please run this script from the streamlit_dashboard directory
    pause
    exit /b 1
)

REM Install dependencies if needed
echo 📦 Installing/updating dependencies...
pip install -r requirements.txt

REM Check configuration
if exist "config.py" (
    echo ✅ Configuration file found
) else (
    echo ⚠️  config.py not found - using default configuration
    echo    Copy config_example.py to config.py and update with your API tokens
)

echo.
echo 🚀 Starting Streamlit application...
echo The dashboard will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the application
echo.

REM Start Streamlit
python -m streamlit run app.py

echo.
echo 👋 Dashboard stopped
pause 