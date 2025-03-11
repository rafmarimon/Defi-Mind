# DEFIMIND Dashboard Deployment Guide

This guide explains how to deploy the DEFIMIND dashboard to your domain (rafaelmarimon.com) while keeping sensitive information protected when sharing the code on GitHub.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Preparing the Environment](#preparing-the-environment)
3. [Deployment Options](#deployment-options)
   - [Option 1: Direct Server Deployment](#option-1-direct-server-deployment)
   - [Option 2: Docker Deployment](#option-2-docker-deployment)
4. [Setting up Nginx as a Reverse Proxy](#setting-up-nginx-as-a-reverse-proxy)
5. [Securing Your Deployment](#securing-your-deployment)
6. [Automated Deployment](#automated-deployment)
7. [GitHub Repository Setup](#github-repository-setup)

## Prerequisites

- A server or VPS with Ubuntu/Debian (recommended)
- Domain name pointing to your server (rafaelmarimon.com)
- SSH access to your server
- Python 3.8+ installed on the server
- Basic knowledge of Linux commands

## Preparing the Environment

1. **Clone your repository** to your server:

```bash
git clone https://github.com/yourusername/defimind.git
cd defimind
```

2. **Set up a Python virtual environment**:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Create production environment file**:

```bash
cp production.env.template production.env
```

4. **Edit the production environment file** with your actual API keys and configuration:

```bash
nano production.env
```

Fill in all the required fields, especially:
- `OPENROUTER_API_KEY`
- `ETHEREUM_RPC_URL`
- `POLYGON_RPC_URL`
- `ALCHEMY_API_KEY`
- `DOMAIN_NAME=rafaelmarimon.com`
- `ENVIRONMENT=production`

## Deployment Options

### Option 1: Direct Server Deployment

1. **Use the deploy script** to run the dashboard:

```bash
python deploy_dashboard.py --host 0.0.0.0 --port 8501 --generate-nginx
```

2. **Set up a systemd service** for automatic startup:

Create a service file:

```bash
sudo nano /etc/systemd/system/defimind.service
```

Add the following content:

```
[Unit]
Description=DEFIMIND Dashboard
After=network.target

[Service]
User=your_username
WorkingDirectory=/path/to/defimind
Environment="ENVIRONMENT=production"
EnvironmentFile=/path/to/defimind/production.env
ExecStart=/path/to/defimind/venv/bin/python deploy_dashboard.py --host 0.0.0.0 --port 8501
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable defimind
sudo systemctl start defimind
```

### Option 2: Docker Deployment

1. **Create a Dockerfile** in your project directory:

```bash
nano Dockerfile
```

Add the following content:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "deploy_dashboard.py", "--host", "0.0.0.0", "--port", "8501"]
```

2. **Create a docker-compose.yml file**:

```bash
nano docker-compose.yml
```

Add the following content:

```yaml
version: '3'
services:
  defimind:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    env_file:
      - production.env
    restart: unless-stopped
```

3. **Build and run the Docker container**:

```bash
docker-compose up -d
```

## Setting up Nginx as a Reverse Proxy

1. **Install Nginx** if not already installed:

```bash
sudo apt update
sudo apt install nginx
```

2. **Copy the generated Nginx configuration**:

```bash
sudo cp nginx_defimind.conf /etc/nginx/sites-available/rafaelmarimon.com
```

3. **Create a symbolic link**:

```bash
sudo ln -s /etc/nginx/sites-available/rafaelmarimon.com /etc/nginx/sites-enabled/
```

4. **Test and restart Nginx**:

```bash
sudo nginx -t
sudo systemctl restart nginx
```

5. **Set up SSL with Let's Encrypt**:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d rafaelmarimon.com
```

## Securing Your Deployment

1. **Enable authentication** by editing your production.env file:

```
DASHBOARD_AUTHENTICATION=true
DASHBOARD_USERNAME=your_username
DASHBOARD_PASSWORD=your_secure_password
```

2. **Set up firewall rules**:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

3. **Regularly update your system**:

```bash
sudo apt update
sudo apt upgrade
```

## Automated Deployment

For automated deployment, consider setting up a CI/CD pipeline with GitHub Actions:

1. **Create a deployment key**:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/defimind_deploy_key
```

2. **Add the public key to your server** as an authorized key.

3. **Add the private key to GitHub secrets** in your repository settings.

4. **Create a GitHub Actions workflow file** `.github/workflows/deploy.yml`:

```yaml
name: Deploy DEFIMIND Dashboard

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Deploy to server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.SERVER_HOST }}
        username: ${{ secrets.SERVER_USERNAME }}
        key: ${{ secrets.DEPLOY_KEY }}
        script: |
          cd /path/to/defimind
          git pull
          source venv/bin/activate
          pip install -r requirements.txt
          sudo systemctl restart defimind
```

## GitHub Repository Setup

When sharing your code on GitHub, make sure to follow these security practices:

1. **Use .gitignore** to exclude sensitive files:
   - The .env file is already in .gitignore
   - production.env should be in .gitignore
   - Any credentials or API keys files

2. **Use environment variables** rather than hardcoded secrets.

3. **Use template files** like `.env.template` and `production.env.template` that show the structure without real values.

4. **Document clearly** in the README that users need to create their own .env file.

5. **Consider using GitHub secrets** for CI/CD workflows rather than committing sensitive values.

By following this guide, you'll have a secure, production-ready DEFIMIND dashboard deployed to rafaelmarimon.com while keeping your sensitive information protected when sharing your codebase on GitHub. 