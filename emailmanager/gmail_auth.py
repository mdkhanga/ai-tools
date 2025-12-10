import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from config import Config


class GmailAuthenticator:
    """Handle Gmail API authentication"""

    def __init__(self):
        self.creds = None

    def authenticate(self):
        """Authenticate and return Gmail service"""
        # Token file stores the user's access and refresh tokens
        if os.path.exists(Config.TOKEN_FILE):
            self.creds = Credentials.from_authorized_user_file(Config.TOKEN_FILE, Config.SCOPES)

        # If there are no (valid) credentials available, let the user log in
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    Config.CREDENTIALS_FILE, Config.SCOPES
                )
                self.creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(Config.TOKEN_FILE, 'w') as token:
                token.write(self.creds.to_json())

        # Build and return the Gmail service
        service = build('gmail', 'v1', credentials=self.creds)
        return service
