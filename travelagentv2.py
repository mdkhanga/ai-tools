#!/usr/bin/python3

import json
import requests
import time
import sys
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Configuration ---
# API key is no longer needed here; the genai.Client() automatically finds it
# from environment variables (GEMINI_API_KEY).
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
MAX_RETRIES = 3

# --- Prompt Templates ---

# System instruction to define the agent's persona and rules.
SYSTEM_INSTRUCTION = """
You are a helpful, iterative travel planning AI. Your goal is to create a multi-day sightseeing itinerary based on the user's location and interests.

You MUST follow these rules strictly:
1. When asked to generate or revise a plan, your entire response MUST be formatted strictly as a single Markdown list.
2. The plan must be structured by day.
3. If the user asks for a revision, analyze the current plan and the user's request, then output the REVISED, COMPLETE plan.
4. **DO NOT ASK FOR MORE INTERESTS** if the user has already provided some input. **ALWAYS** generate the plan based on the data available.
5. Keep the plan concise and focused on major tourist sights and activities.
"""

# The initial prompt template
INITIAL_PLAN_PROMPT = """
Please generate a detailed, day-by-day sightseeing itinerary for a trip to {destination} from {start_date} to {end_date}.
The user's stated interests are: {interests}.
Provide the full plan now.
"""

# The iterative prompt template for revisions
REVISION_PLAN_PROMPT = """
The user has provided feedback on the existing itinerary.
Current Itinerary:
---
{current_plan}
---
User Feedback/Request: "{user_feedback}"

Please REVISE the entire itinerary based on the user's feedback. Output only the revised, complete plan using the required Markdown list format.
"""

class TravelAgent:
    """
    An interactive, stateful AI agent built in plain Python to manage a travel plan.
    Uses the google-genai SDK for API calls.
    """
    def __init__(self, destination, start_date, end_date):
        self.destination = destination
        self.start_date = start_date
        self.end_date = end_date
        self.interests = None
        self.current_plan = None
        
        # Initialize the Gemini client
        try:
            self.client = genai.Client()
        except Exception as e:
            print(f"Error initializing Gemini client: {e}", file=sys.stderr)
            print("Please ensure the GEMINI_API_KEY environment variable is set.", file=sys.stderr)
            sys.exit(1)
        
        # Memory structure: Stores conversation history for context (will be types.Content list)
        self.chat_history: list[types.Content] = []
        
        # NOTE: System instructions are now passed via configuration, not in chat_history,
        # but we use self.system_instruction to store the prompt content.
        self.system_instruction = SYSTEM_INSTRUCTION


    def _call_gemini(self, user_prompt, temperature=0.7):
        """Generic function to call the Gemini API using genai.Client with exponential backoff."""

        print("User prompt is :" + user_prompt)
        
        # CORRECTED: Pass the text string directly to the Part object
        # The constructor for Part.from_text() takes one string argument.
        new_user_content = types.Content(role="user", parts=[types.Part(text=user_prompt)])
        self.chat_history.append(new_user_content)

        # Configure the request
        config = types.GenerateContentConfig(
            temperature=temperature,
            # The system instruction is passed outside the contents list
            system_instruction=self.system_instruction
        )
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=self.chat_history,
                    config=config,
                )
                
                # Check for empty response or blocked content
                if not response.candidates or not response.candidates[0].content.parts:
                    print(f"API Error: Did not receive valid content in response.", file=sys.stderr)
                    # We might get a response but no generated text if it was blocked or errored
                    return "Sorry, the AI could not generate a response. Please try again."

                text = response.text
                
                # Store the model's response in history for the next turn
                # CORRECTED: Pass the text string directly to the Part object
                model_content = types.Content(role="model", parts=[types.Part(text=text)])
                self.chat_history.append(model_content)
                
                return text

            except APIError as e:
                print(f"API Call failed (Attempt {attempt + 1}/{MAX_RETRIES}): {e}", file=sys.stderr)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    return "Error: Unable to connect to the AI service after multiple retries."
            except Exception as e:
                # Catch other potential errors (e.g., JSON parsing if using raw fetch, though less likely with SDK)
                print(f"An unexpected error occurred (Attempt {attempt + 1}/{MAX_RETRIES}): {e}", file=sys.stderr)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    return "Critical Error: An unrecoverable error occurred."

        # Fallback response if all retries fail
        return "Critical Error: The AI service is currently unavailable."


    def _get_initial_plan(self):
        """Generates the first itinerary."""
        prompt = INITIAL_PLAN_PROMPT.format(
            destination=self.destination,
            start_date=self.start_date,
            end_date=self.end_date,
            interests=self.interests
        )
        print("\n[AGENT] Generating initial plan based on your interests...")
        return self._call_gemini(prompt)

    def _update_plan(self, user_feedback):
        """Revises the itinerary based on user feedback."""
        # We need to construct a prompt that includes the current plan for the LLM to read.
        # Since the plan itself is often long and formatted (Markdown list),
        # we construct the revision prompt as the user input for the current turn.
        
        revision_prompt = REVISION_PLAN_PROMPT.format(
            current_plan=self.current_plan,
            user_feedback=user_feedback
        )
        
        print("\n[AGENT] Revising plan based on your feedback...")
        return self._call_gemini(revision_prompt)
    
    def run_agent(self):
        """The main interactive loop for the travel agent."""
        print("-" * 50)
        print(f"🌍 Travel Agent Initializing for: {self.destination}")
        print(f"🗓️ Dates: {self.start_date} to {self.end_date}")
        print("-" * 50)
        
        # --- PHASE 1: Gather Missing Inputs ---
        
        # Start by asking for interests
        print("[AGENT] Before I can build a plan, what kinds of things are you interested in?")
        print("e.g., History, hiking, local food, museums, nightlife.")
        
        while self.interests is None:
            user_input = input("YOU: ")
            if user_input.strip():
                self.interests = user_input.strip()
            else:
                print("[AGENT] Please tell me your interests.")

        # --- PHASE 2: Generate Initial Plan ---
        
        new_plan = self._get_initial_plan()
        
        # Check for error state from API call
        if not new_plan.startswith("Error"):
            self.current_plan = new_plan
            print("\n" * 2)
            print("=" * 50)
            print("🗺️ INITIAL ITINERARY GENERATED 🗺️")
            print("=" * 50)
            print(self.current_plan)
            print("=" * 50)
        else:
            print(f"\n[CRITICAL ERROR]: {new_plan}")
            return


        # --- PHASE 3: Iterative Revision Loop ---
        print("\n[AGENT] How does this look? You can ask me to change anything.")
        print("Example: 'Replace Day 2 activity with something focused on local food' or 'Remove all history items.'")
        print("Type 'exit' to finish.")

        while True:
            user_input = input("YOU: ")
            if user_input.lower() in ["exit", "quit", "thank you"]:
                print("\n[AGENT] Thank you for planning with me! Enjoy your trip!")
                break
            
            if not self.current_plan:
                print("[AGENT] Error: Current plan is missing. Restarting process.", file=sys.stderr)
                break
            
            # Use the LLM to process the user feedback and regenerate the plan
            revised_plan = self._update_plan(user_input)

            if not revised_plan.startswith("Error"):
                # Update the state (memory) with the new plan
                # Note: We overwrite the plan state, but the full interaction history remains in self.chat_history
                self.current_plan = revised_plan 
                print("\n" * 2)
                print("=" * 50)
                print("🔄 REVISED ITINERARY 🔄")
                print("=" * 50)
                print(self.current_plan)
                print("=" * 50)
            else:
                print(f"\n[ERROR]: {revised_plan}")
                
        # Clear sensitive data from memory when done
        self.chat_history = []
        self.current_plan = None
        self.interests = None


# --- Execution Block ---
if __name__ == "__main__":
    # --- Simulated User Inputs ---
    DESTINATION = "Rome, Italy"
    START_DATE = "October 10, 2025"
    END_DATE = "October 14, 2025"
    
    # Note: The SDK automatically looks for the GEMINI_API_KEY environment variable.
    # No need for manual key handling here.

    try:
        agent = TravelAgent(DESTINATION, START_DATE, END_DATE)
        agent.run_agent()
    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}", file=sys.stderr)