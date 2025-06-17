import base64
import os
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import pickle

SCOPES = ["https://mail.google.com/"]
TOKEN = Path("token.json")
CREDS = Path("client_secret_270395908102-4nasug7tjig4imq3bp67cj2tktgirq58.apps.googleusercontent.com.json")
OUTPUT_DIR = Path("chess_emails")

def get_gmail_service():
    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDS, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN, 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def get_label_id(service, label_name):
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])
    for label in labels:
        if label['name'] == label_name:
            return label['id']
    return None

def download_emails():
    service = get_gmail_service()
    label_id = get_label_id(service, "Chess Games")
    
    if not label_id:
        print("Label 'Chess Games' not found")
        return
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Get all messages with the label (handle pagination)
    messages = []
    page_token = None
    while True:
        response = service.users().messages().list(
            userId='me',
            labelIds=[label_id],
            maxResults=500,  # Gmail API max per request
            pageToken=page_token
        ).execute()

        messages.extend(response.get('messages', []))

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    if not messages:
        print("No messages found.")
        return

    print(f"Found {len(messages)} messages. Downloading...")
    
    for i, message in enumerate(messages, 1):
        msg = service.users().messages().get(
            userId='me',
            id=message['id'],
            format='full'
        ).execute()
        
        # Get email subject
        headers = msg['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        
        # Get email body
        if 'parts' in msg['payload']:
            parts = msg['payload']['parts']
            body = ''
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode()
                    break
        else:
            body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode()
        
        # Save to file
        filename = f"email_{i:04d}_{subject[:50]}.txt"
        filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.'))
        filepath = OUTPUT_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Subject: {subject}\n\n")
            f.write(body)
        
        if i % 100 == 0:
            print(f"Downloaded {i} emails...")

if __name__ == '__main__':
    download_emails() 