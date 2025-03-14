# DEFIMIND Digital Ocean Deployment

This directory contains the necessary files for deploying DEFIMIND to Digital Ocean App Platform.

## Files

- `requirements.txt`: A minimal set of dependencies for deployment
- `app.yaml`: The Digital Ocean App Platform configuration file

## Deployment Instructions

### Using Digital Ocean Dashboard

1. Log in to your Digital Ocean account
2. Navigate to "Apps" and click "Create App"
3. Select "GitHub" as your source
4. Connect to your GitHub account and select the repository
5. Select the `main` branch
6. Configure environment variables (see below)
7. Choose "Basic" plan or higher
8. Deploy the app

### Using Digital Ocean CLI

1. Install the Digital Ocean CLI:
   ```
   brew install doctl
   ```

2. Authenticate with your API token:
   ```
   doctl auth init
   ```

3. Create the app from the YAML file:
   ```
   doctl apps create --spec deploy_files/app.yaml
   ```

## Environment Variables

You'll need to configure the following environment variables in your Digital Ocean App:

- `OPENAI_API_KEY`: Your OpenAI API key
- `ETHERSCAN_API_KEY`: Your Etherscan API key (optional)
- `INFURA_API_KEY`: Your Infura API key (optional)
- `ALCHEMY_API_KEY`: Your Alchemy API key (optional)

## Troubleshooting

### "No space left on device" error

If you encounter a "no space left on device" error during deployment:

1. Increase the instance size in `app.yaml` to a larger size (e.g., professional-s)
2. Ensure you're using the minimal `requirements.txt`
3. Consider removing any unnecessary dependencies

### Other Issues

For other deployment issues, check the Digital Ocean logs by:

1. Going to your app in the Digital Ocean dashboard
2. Clicking on "Components" 
3. Selecting your component
4. Viewing the logs

You can also stream logs via the CLI:
```
doctl apps logs [APP_ID] --component-name defimind-app
``` 