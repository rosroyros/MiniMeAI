import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow

GMAIL_CREDENTIALS_FILE = "/app/config/gmail_credentials.json"
GMAIL_TOKEN_FILE = "/app/config/gmail_token.pickle"
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

print(f"Using credentials file: {GMAIL_CREDENTIALS_FILE}")

# Get new credentials
flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, SCOPES)
creds = flow.run_local_server(port=8080, host='0.0.0.0')

# Save credentials
with open(GMAIL_TOKEN_FILE, "wb") as token:
    pickle.dump(creds, token)

print(f"Token saved to {GMAIL_TOKEN_FILE}")
