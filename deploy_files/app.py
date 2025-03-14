#!/usr/bin/env python3
"""
DEFIMIND Application Entry Point

This file serves as the main entry point for Digital Ocean App Platform.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("defimind-app")

# Ensure we can import from core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def start_dashboard():
    """
    Start the Streamlit dashboard
    """
    import streamlit.web.bootstrap as bootstrap
    from streamlit.web.server import Server
    
    # Get port from environment (Digital Ocean expects 8080)
    port = int(os.environ.get("PORT", 8080))
    
    # Set Streamlit port
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    
    logger.info(f"Starting DEFIMIND dashboard on port {port}")
    
    # Look for the dashboard file in the deploy_files directory
    dashboard_file = os.path.join("deploy_files", "dashboard.py")
    
    if os.path.exists(dashboard_file):
        logger.info(f"Found dashboard file: {dashboard_file}")
        
        # Start Streamlit dashboard
        bootstrap._configure_logger()
        flag_options = {
            "server.port": port,
            "server.headless": True,
            "browser.serverAddress": "0.0.0.0",
            "global.developmentMode": False,
        }
        bootstrap._set_up_extra_flags(flag_options=flag_options)
        Server.get_current()._is_running_hello = False
        bootstrap._run_script(dashboard_file, "", [], flag_options=flag_options)
    else:
        # If no dashboard file is found, create a simple Streamlit app
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"""
import streamlit as st

st.title("DEFIMIND")
st.header("Welcome to DEFIMIND DeFi Analytics Platform")
st.write("The dashboard is being configured. Please check back soon.")
"""
            )
            temp_script = f.name
        
        # Start Streamlit with the temporary script
        bootstrap._configure_logger()
        flag_options = {
            "server.port": port,
            "server.headless": True,
            "browser.serverAddress": "0.0.0.0",
            "global.developmentMode": False,
        }
        bootstrap._set_up_extra_flags(flag_options=flag_options)
        Server.get_current()._is_running_hello = False
        bootstrap._run_script(temp_script, "", [], flag_options=flag_options)
            
if __name__ == "__main__":
    start_dashboard() 