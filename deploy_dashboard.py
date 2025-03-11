#!/usr/bin/env python3
"""
DEFIMIND Dashboard Deployment Script

This script helps deploy the DEFIMIND dashboard to a production environment.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))
DEFAULT_BASE_URL = os.getenv("DASHBOARD_BASE_URL", "")

def validate_environment():
    """Validate that we have the necessary environment variables set"""
    required_vars = [
        "OPENROUTER_API_KEY",
        "ETHEREUM_RPC_URL",
        "POLYGON_RPC_URL",
        "ALCHEMY_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment")
        return False
    
    return True

def ensure_dependencies():
    """Ensure all required dependencies are installed"""
    required_packages = [
        "streamlit",
        "langchain",
        "langchain-openai"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    return True

def deploy_streamlit(host, port, base_url):
    """Deploy the Streamlit app"""
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        return False
    
    cmd = [
        "streamlit", "run", str(dashboard_path),
        "--server.address", host,
        "--server.port", str(port)
    ]
    
    if base_url:
        cmd.extend(["--server.baseUrlPath", base_url])
    
    # Run in the foreground
    print(f"Deploying dashboard on {host}:{port}" + (f" with base URL {base_url}" if base_url else ""))
    subprocess.run(cmd)
    
    return True

def setup_nginx_config(domain, port):
    """Generate an nginx configuration for the domain"""
    config = f"""
server {{
    listen 80;
    listen [::]:80;
    server_name {domain};
    
    location / {{
        proxy_pass http://localhost:{port};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }}
}}
"""
    
    config_path = Path("nginx_defimind.conf")
    with open(config_path, "w") as f:
        f.write(config)
    
    print(f"Generated nginx configuration at {config_path}")
    print("To use this configuration with nginx:")
    print(f"1. Copy it to /etc/nginx/sites-available/{domain}")
    print(f"2. Create a symlink: sudo ln -s /etc/nginx/sites-available/{domain} /etc/nginx/sites-enabled/")
    print("3. Test the configuration: sudo nginx -t")
    print("4. Reload nginx: sudo systemctl reload nginx")
    
    return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy DEFIMIND Dashboard")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host address to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run the dashboard on")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL path for the dashboard")
    parser.add_argument("--domain", default="rafaelmarimon.com", help="Domain name for nginx configuration")
    parser.add_argument("--generate-nginx", action="store_true", help="Generate nginx configuration")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Validate environment
    if not validate_environment():
        return 1
    
    # Ensure dependencies are installed
    if not ensure_dependencies():
        return 1
    
    # Generate nginx configuration if requested
    if args.generate_nginx:
        if not setup_nginx_config(args.domain, args.port):
            return 1
    
    # Deploy the dashboard
    if not deploy_streamlit(args.host, args.port, args.base_url):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 