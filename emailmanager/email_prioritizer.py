from typing import TypedDict, List, Dict
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from gmail_tools import get_gmail_service
from config import Config
import base64
import json


class PrioritizerState(TypedDict):
    """State for email prioritization workflow"""
    emails: List[Dict]  # Raw emails from Gmail
    analyzed_emails: List[Dict]  # Emails with importance scores
    top_emails: List[Dict]  # Top N most important emails
    summary: str  # Summary of important emails


class EmailPrioritizer:
    """
    LangGraph workflow for identifying important emails.

    This workflow:
    1. Fetches recent unread emails
    2. Analyzes each email for importance using AI
    3. Ranks emails by importance
    4. Returns top N most important emails with summary
    """

    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.3  # Lower temperature for more consistent analysis
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the email prioritization workflow"""
        workflow = StateGraph(PrioritizerState)

        # Add nodes
        workflow.add_node("fetch_emails", self._fetch_emails)
        workflow.add_node("analyze_importance", self._analyze_importance)
        workflow.add_node("rank_emails", self._rank_emails)
        workflow.add_node("generate_summary", self._generate_summary)

        # Set entry point and edges
        workflow.set_entry_point("fetch_emails")
        workflow.add_edge("fetch_emails", "analyze_importance")
        workflow.add_edge("analyze_importance", "rank_emails")
        workflow.add_edge("rank_emails", "generate_summary")
        workflow.add_edge("generate_summary", END)

        return workflow.compile()

    def _fetch_emails(self, state: PrioritizerState):
        """Fetch recent unread emails from Gmail"""
        try:
            service = get_gmail_service()
            results = service.users().messages().list(
                userId='me',
                q='is:unread in:inbox',
                maxResults=20
            ).execute()

            messages = results.get('messages', [])
            emails = []

            for msg in messages:
                email_data = service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='full'
                ).execute()

                headers = {h['name']: h['value'] for h in email_data['payload']['headers']}

                # Get email body
                body = ""
                if 'parts' in email_data['payload']:
                    for part in email_data['payload']['parts']:
                        if part['mimeType'] == 'text/plain':
                            if 'data' in part['body']:
                                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                                break
                else:
                    if 'body' in email_data['payload'] and 'data' in email_data['payload']['body']:
                        body = base64.urlsafe_b64decode(email_data['payload']['body']['data']).decode('utf-8', errors='ignore')

                emails.append({
                    'id': msg['id'],
                    'from': headers.get('From', 'Unknown'),
                    'subject': headers.get('Subject', 'No Subject'),
                    'date': headers.get('Date', 'Unknown'),
                    'snippet': email_data.get('snippet', ''),
                    'body': body[:500] if body else email_data.get('snippet', '')  # Limit body length
                })

            return {"emails": emails}

        except Exception as e:
            print(f"Error fetching emails: {e}")
            return {"emails": []}

    def _analyze_importance(self, state: PrioritizerState):
        """Analyze importance of each email using AI"""
        analyzed_emails = []

        for email in state['emails']:
            prompt = f"""Analyze the importance of this email on a scale of 1-10.

Consider:
- Sender authority/importance
- Subject urgency
- Content requiring action
- Time sensitivity
- Business/personal relevance

Email Details:
From: {email['from']}
Subject: {email['subject']}
Date: {email['date']}
Preview: {email['snippet']}

Respond with ONLY a JSON object in this format:
{{"importance_score": <number 1-10>, "reason": "<brief explanation>"}}"""

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                # Parse the response
                response_text = response.content.strip()

                # Extract JSON from response (handle markdown code blocks)
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()

                analysis = json.loads(response_text)

                email_with_score = email.copy()
                email_with_score['importance_score'] = analysis.get('importance_score', 5)
                email_with_score['importance_reason'] = analysis.get('reason', 'No reason provided')

                analyzed_emails.append(email_with_score)

            except Exception as e:
                print(f"Error analyzing email {email['id']}: {e}")
                # Default score if analysis fails
                email_with_score = email.copy()
                email_with_score['importance_score'] = 5
                email_with_score['importance_reason'] = 'Analysis failed'
                analyzed_emails.append(email_with_score)

        return {"analyzed_emails": analyzed_emails}

    def _rank_emails(self, state: PrioritizerState):
        """Rank emails by importance and select top N"""
        ranked = sorted(
            state['analyzed_emails'],
            key=lambda x: x['importance_score'],
            reverse=True
        )

        top_emails = ranked[:self.top_n]
        return {"top_emails": top_emails}

    def _generate_summary(self, state: PrioritizerState):
        """Generate a summary of the most important emails"""
        if not state['top_emails']:
            return {"summary": "No important emails found."}

        emails_text = "\n\n".join([
            f"{i+1}. From: {email['from']}\n   Subject: {email['subject']}\n   Importance: {email['importance_score']}/10\n   Reason: {email['importance_reason']}"
            for i, email in enumerate(state['top_emails'])
        ])

        prompt = f"""Provide a concise executive summary of these important emails:

{emails_text}

Summary should be 2-3 sentences highlighting key actions needed."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"summary": response.content}

    def get_top_important_emails(self):
        """
        Get the top N most important emails of the day.

        Returns:
            Dictionary with top emails and summary
        """
        initial_state = {
            "emails": [],
            "analyzed_emails": [],
            "top_emails": [],
            "summary": ""
        }

        result = self.graph.invoke(initial_state)

        return {
            "top_emails": result['top_emails'],
            "summary": result['summary'],
            "total_analyzed": len(result['emails'])
        }
