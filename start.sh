#!/bin/bash

# Install Playwright browser dependencies
echo "Installing Playwright dependencies..."
python -m playwright install --with-deps chromium

# Start the application
echo "Starting DEFIMIND..."
exec python app.py 