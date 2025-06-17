from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://mail.google.com/"]          # full read/write; reduce later if desired
TOKEN   = Path("token.json")                   # stores refresh+access tokens
CREDS   = Path("credentials.json")             # downloaded from Google Cloud console

def gmail_service():
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    else:
        flow  = InstalledAppFlow.from_client_secrets_file(CREDS, SCOPES)
        creds = flow.run_local_server(port=0)  # opens a browser once
        TOKEN.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds, cache_discovery=False)
