import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for Gmail AI Agent"""

    # Gmail API settings
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.modify',
        'https://www.googleapis.com/auth/gmail.compose'
    ]

    # Gemini API key
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # Token file for Gmail OAuth
    TOKEN_FILE = 'token.json'
    CREDENTIALS_FILE = 'credentials.json'

    @classmethod
    def validate(cls):
        """Validate that all required configuration is present"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        if not os.path.exists(cls.CREDENTIALS_FILE):
            raise FileNotFoundError(
                f"{cls.CREDENTIALS_FILE} not found. Please download it from Google Cloud Console."
            )
