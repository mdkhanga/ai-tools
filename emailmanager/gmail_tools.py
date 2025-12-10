import base64
from email.mime.text import MIMEText
from typing import Optional, List, Dict
from langchain_core.tools import tool
from gmail_auth import GmailAuthenticator


# Initialize Gmail service globally
authenticator = GmailAuthenticator()
gmail_service = None


def get_gmail_service():
    """Lazy initialization of Gmail service"""
    global gmail_service
    if gmail_service is None:
        gmail_service = authenticator.authenticate()
    return gmail_service


@tool
def search_emails(query: str, max_results: int = 10) -> str:
    """
    Search for emails using Gmail search syntax.

    Args:
        query: Gmail search query (e.g., 'from:user@example.com', 'subject:meeting', 'is:unread')
        max_results: Maximum number of results to return (default: 10)

    Returns:
        JSON string with email results including id, threadId, and snippet
    """
    try:
        service = get_gmail_service()
        results = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()

        messages = results.get('messages', [])

        if not messages:
            return "No emails found matching the query."

        email_list = []
        for msg in messages:
            email_data = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['From', 'Subject', 'Date']
            ).execute()

            headers = {h['name']: h['value'] for h in email_data['payload']['headers']}

            email_list.append({
                'id': msg['id'],
                'threadId': msg['threadId'],
                'from': headers.get('From', 'Unknown'),
                'subject': headers.get('Subject', 'No Subject'),
                'date': headers.get('Date', 'Unknown'),
                'snippet': email_data.get('snippet', '')
            })

        return str(email_list)
    except Exception as e:
        return f"Error searching emails: {str(e)}"


@tool
def read_email(email_id: str) -> str:
    """
    Read the full content of an email by its ID.

    Args:
        email_id: The Gmail message ID

    Returns:
        Full email content including headers and body
    """
    try:
        service = get_gmail_service()
        message = service.users().messages().get(
            userId='me',
            id=email_id,
            format='full'
        ).execute()

        headers = {h['name']: h['value'] for h in message['payload']['headers']}

        # Get email body
        body = ""
        if 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
        else:
            if 'body' in message['payload'] and 'data' in message['payload']['body']:
                body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')

        result = {
            'id': email_id,
            'from': headers.get('From', 'Unknown'),
            'to': headers.get('To', 'Unknown'),
            'subject': headers.get('Subject', 'No Subject'),
            'date': headers.get('Date', 'Unknown'),
            'body': body or message.get('snippet', 'No content available')
        }

        return str(result)
    except Exception as e:
        return f"Error reading email: {str(e)}"


@tool
def compose_and_send_email(to: str, subject: str, body: str) -> str:
    """
    Compose and send a new email.

    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body content

    Returns:
        Success message with sent message ID
    """
    try:
        service = get_gmail_service()

        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')

        send_message = service.users().messages().send(
            userId='me',
            body={'raw': raw_message}
        ).execute()

        return f"Email sent successfully! Message ID: {send_message['id']}"
    except Exception as e:
        return f"Error sending email: {str(e)}"


@tool
def reply_to_email(email_id: str, reply_body: str) -> str:
    """
    Reply to an existing email.

    Args:
        email_id: The Gmail message ID to reply to
        reply_body: The reply message content

    Returns:
        Success message with sent reply ID
    """
    try:
        service = get_gmail_service()

        # Get original message to extract headers
        original = service.users().messages().get(
            userId='me',
            id=email_id,
            format='metadata',
            metadataHeaders=['From', 'Subject', 'Message-ID']
        ).execute()

        headers = {h['name']: h['value'] for h in original['payload']['headers']}

        # Create reply
        message = MIMEText(reply_body)
        message['to'] = headers.get('From')
        subject = headers.get('Subject', '')
        if not subject.startswith('Re: '):
            subject = 'Re: ' + subject
        message['subject'] = subject
        message['In-Reply-To'] = headers.get('Message-ID')
        message['References'] = headers.get('Message-ID')

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')

        send_message = service.users().messages().send(
            userId='me',
            body={
                'raw': raw_message,
                'threadId': original['threadId']
            }
        ).execute()

        return f"Reply sent successfully! Message ID: {send_message['id']}"
    except Exception as e:
        return f"Error replying to email: {str(e)}"


@tool
def delete_email(email_id: str) -> str:
    """
    Delete an email (move to trash).

    Args:
        email_id: The Gmail message ID to delete

    Returns:
        Success or error message
    """
    try:
        service = get_gmail_service()
        service.users().messages().trash(userId='me', id=email_id).execute()
        return f"Email {email_id} moved to trash successfully."
    except Exception as e:
        return f"Error deleting email: {str(e)}"


@tool
def get_unread_emails(max_results: int = 20) -> str:
    """
    Get unread emails from inbox.

    Args:
        max_results: Maximum number of unread emails to retrieve (default: 20)

    Returns:
        List of unread emails with basic information
    """
    return search_emails(query="is:unread in:inbox", max_results=max_results)


# List of all tools for the agent
gmail_tools = [
    search_emails,
    read_email,
    compose_and_send_email,
    reply_to_email,
    delete_email,
    get_unread_emails
]
