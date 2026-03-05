#!/usr/bin/env python3
"""
Gmail AI Agent - Main CLI Interface

This demonstrates multiple agentic patterns:
1. ReAct (Reasoning + Acting) - for general email operations
2. Reflection - for email composition with self-critique
3. Multi-step workflows - for email prioritization
"""

import sys
from config import Config
from react_agent import EmailReActAgent
from reflection_composer import ReflectionEmailComposer
from email_prioritizer import EmailPrioritizer


def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("         Gmail AI Agent - Powered by LangGraph & Gemini")
    print("=" * 70)
    print("\nDemonstrating Agentic Patterns:")
    print("  • ReAct (Reasoning + Acting) for email operations")
    print("  • Reflection for email composition")
    print("  • Multi-agent workflows for prioritization")
    print("=" * 70)
    print()


def print_menu():
    """Print main menu"""
    print("\n" + "=" * 70)
    print("Available Commands:")
    print("=" * 70)
    print("1. chat          - Chat with the ReAct email agent")
    print("2. important     - Get today's 3 most important emails (AI prioritization)")
    print("3. compose       - Compose email with reflection pattern")
    print("4. search        - Quick search emails")
    print("5. help          - Show this menu")
    print("6. quit          - Exit the application")
    print("=" * 70)


def chat_mode():
    """Interactive chat with ReAct agent"""
    print("\n🤖 ReAct Agent Mode (Reasoning + Acting)")
    print("The agent will reason about your request and take appropriate actions.")
    print("Type 'back' to return to main menu.\n")

    agent = EmailReActAgent()

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['back', 'exit', 'quit']:
            break

        if not user_input:
            continue

        print("\n🧠 Agent is thinking and acting...\n")

        try:
            response = agent.run(user_input)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Error: {e}")


def get_important_emails():
    """Get top 3 important emails using AI prioritization"""
    print("\n📧 Analyzing your emails to find the most important ones...")
    print("This uses a multi-step LangGraph workflow:\n")
    print("  1. Fetching last 10 unread emails")
    print("  2. Analyzing importance with AI (batch call)")
    print("  3. Ranking by priority")
    print("  4. Generating summary\n")

    try:
        prioritizer = EmailPrioritizer(top_n=3)
        result = prioritizer.get_top_important_emails()

        print(f"\n✨ Analyzed {result['total_analyzed']} emails\n")
        print("=" * 70)
        print("TOP 3 MOST IMPORTANT EMAILS")
        print("=" * 70)

        if not result['top_emails']:
            print("\nNo unread emails found!")
            return

        for i, email in enumerate(result['top_emails'], 1):
            print(f"\n{i}. {email['subject']}")
            print(f"   From: {email['from']}")
            print(f"   Date: {email['date']}")
            print(f"   Importance: {email['importance_score']}/10")
            print(f"   Why: {email['importance_reason']}")
            print(f"   Preview: {email['snippet'][:100]}...")

        print("\n" + "=" * 70)
        print("EXECUTIVE SUMMARY")
        print("=" * 70)
        print(f"\n{result['summary']}")
        print()

    except Exception as e:
        print(f"Error: {e}")


def compose_with_reflection():
    """Compose email using reflection pattern"""
    print("\n✍️  Reflection-based Email Composer")
    print("The AI will draft, critique, and revise the email automatically.\n")

    request = input("Describe the email you want to compose: ").strip()

    if not request:
        print("No request provided.")
        return

    print("\n🔄 Composing with reflection pattern...")
    print("  1. Generating draft with built-in reflection")
    print("  2. Finalizing\n")
    print("=" * 70)
    print("LLM CALL TRACKING")
    print("=" * 70)

    try:
        composer = ReflectionEmailComposer(max_revisions=1)
        result = composer.compose(request)

        print("=" * 70)

        print("=" * 70)
        print("FINAL EMAIL (after reflection)")
        print("=" * 70)
        print(f"\n{result['final_email']}\n")
        print("=" * 70)
        print(f"Revisions made: {result['revisions_made']}")
        print(f"Final assessment: {result['final_critique']}")
        print("=" * 70)

        # Ask if user wants to send it
        send = input("\n📤 Would you like to send this email? (yes/no): ").strip().lower()

        if send == 'yes':
            to = input("Recipient email: ").strip()
            subject = input("Subject: ").strip()

            if to and subject:
                print("\n⚠️  To actually send, integrate with compose_and_send_email tool")
                print(f"Would send to: {to}")
                print(f"Subject: {subject}")
            else:
                print("Missing recipient or subject. Email not sent.")

    except Exception as e:
        print(f"Error: {e}")


def quick_search():
    """Quick email search"""
    print("\n🔍 Quick Search")
    query = input("Enter search query (Gmail syntax): ").strip()

    if not query:
        print("No query provided.")
        return

    print(f"\n⚠️  To implement: Use the ReAct agent in chat mode")
    print(f"Try: 'chat' then ask 'search for {query}'")


def main():
    """Main application loop"""
    print_banner()

    # Validate configuration
    try:
        Config.validate()
        print("✅ Configuration validated")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease ensure:")
        print("1. Create a .env file with GEMINI_API_KEY")
        print("2. Download credentials.json from Google Cloud Console")
        print("   (Enable Gmail API and create OAuth 2.0 credentials)")
        sys.exit(1)

    print_menu()

    while True:
        command = input("\nEnter command: ").strip().lower()

        if command in ['1', 'chat']:
            chat_mode()

        elif command in ['2', 'important']:
            get_important_emails()

        elif command in ['3', 'compose']:
            compose_with_reflection()

        elif command in ['4', 'search']:
            quick_search()

        elif command in ['5', 'help']:
            print_menu()

        elif command in ['6', 'quit', 'exit']:
            print("\n👋 Goodbye!")
            sys.exit(0)

        else:
            print(f"Unknown command: {command}")
            print("Type 'help' to see available commands.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
