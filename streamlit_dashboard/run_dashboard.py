#!/usr/bin/env python3
"""
Startup script for UbiOps Server Usage Dashboard
This script checks dependencies and launches the Streamlit application.
"""

import sys
import subprocess
import importlib.util

def check_dependency(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_dependency(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 UbiOps Server Usage Dashboard - Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    import os
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the streamlit_dashboard directory")
        sys.exit(1)
    
    # Check required dependencies
    required_packages = [
        "streamlit",
        "ubiops", 
        "pandas",
        "plotly",
        "python-dateutil"
    ]
    
    missing_packages = []
    
    print("📦 Checking dependencies...")
    for package in required_packages:
        if check_dependency(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"\n📥 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_dependency(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
                sys.exit(1)
    
    # Check configuration
    print("\n⚙️  Checking configuration...")
    if os.path.exists("config.py"):
        print("✅ config.py found")
    else:
        print("⚠️  config.py not found - using default configuration")
        print("   Copy config_example.py to config.py and update with your API tokens")
    
    # Launch Streamlit
    print("\n🚀 Starting Streamlit application...")
    print("The dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 