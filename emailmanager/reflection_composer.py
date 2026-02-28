from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global counter for LLM calls
llm_call_counter = 0


class EmailDraftState(TypedDict):
    """State for email composition with reflection"""
    request: str  # Original user request
    draft: str  # Current email draft
    critique: str  # Critique of the draft
    revision_count: int  # Number of revisions made
    max_revisions: int  # Maximum revisions allowed
    final_email: str  # Final approved email


class ReflectionEmailComposer:
    """
    Email composer using the reflection pattern.

    This demonstrates the reflection agentic pattern:
    1. Generate: Create an initial draft
    2. Reflect: Critically evaluate the draft
    3. Revise: Improve based on reflection
    4. Repeat until satisfactory or max iterations reached
    """

    def __init__(self, max_revisions: int = 1):
        self.max_revisions = max_revisions
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.7
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the reflection workflow graph"""
        workflow = StateGraph(EmailDraftState)

        # Add nodes
        workflow.add_node("generate_draft", self._generate_draft)
        workflow.add_node("reflect", self._reflect_on_draft)
        workflow.add_node("revise", self._revise_draft)
        workflow.add_node("finalize", self._finalize)

        # Set entry point
        workflow.set_entry_point("generate_draft")

        # Add edges
        workflow.add_edge("generate_draft", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self._should_revise,
            {
                "revise": "revise",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("revise", "reflect")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _generate_draft(self, state: EmailDraftState):
        """Generate initial email draft with self-reflection in one call"""
        global llm_call_counter
        llm_call_counter += 1
        logger.info(f"🔵 LLM CALL #{llm_call_counter} - reflection_composer._generate_draft()")
        print(f"🔵 LLM CALL #{llm_call_counter} - Generating email with built-in reflection")

        prompt = f"""Generate a professional email based on this request, then critique and improve it.

Request: {state['request']}

Step 1: Write an initial draft email with:
- Appropriate greeting
- Clear subject line suggestion
- Well-structured body
- Professional closing

Step 2: Self-critique the draft on:
- Clarity and conciseness
- Professional tone
- Grammar and spelling
- Completeness

Step 3: Provide the final improved version.

Respond ONLY with the final improved email (not the critique or initial draft)."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "draft": response.content,
            "revision_count": 0,
            "max_revisions": self.max_revisions,
            "critique": "Generated with built-in reflection"
        }

    def _reflect_on_draft(self, state: EmailDraftState):
        """Skip explicit reflection - already done during generation"""
        print(f"📝 Reflecting on draft (no LLM call - returning APPROVED)")
        return {"critique": "APPROVED - Generated with built-in reflection"}

    def _should_revise(self, state: EmailDraftState):
        """Decide whether to revise or finalize"""
        # If approved or max revisions reached, finalize
        if "APPROVED" in state['critique'].upper() or state['revision_count'] >= state['max_revisions']:
            return "finalize"
        return "revise"

    def _revise_draft(self, state: EmailDraftState):
        """Revise the draft based on critique"""
        global llm_call_counter
        llm_call_counter += 1
        logger.warning(f"⚠️ LLM CALL #{llm_call_counter} - reflection_composer._revise_draft() - THIS SHOULD NOT BE CALLED!")
        print(f"⚠️ LLM CALL #{llm_call_counter} - Revising draft (unexpected!)")

        prompt = f"""Revise this email draft based on the critique provided.

Original Request: {state['request']}

Current Draft:
{state['draft']}

Critique:
{state['critique']}

Provide an improved version of the email that addresses the critique while maintaining professionalism.

Revised Email:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "draft": response.content,
            "revision_count": state['revision_count'] + 1
        }

    def _finalize(self, state: EmailDraftState):
        """Finalize the email"""
        print(f"✅ Finalizing email (no LLM call)")
        global llm_call_counter
        print(f"\n📊 TOTAL LLM CALLS FOR COMPOSE: {llm_call_counter}")
        return {"final_email": state['draft']}

    def compose(self, request: str):
        """
        Compose an email with reflection.

        Args:
            request: User's request describing the email to compose

        Returns:
            Dictionary with final_email and reflection history
        """
        # Reset LLM call counter for this compose operation
        global llm_call_counter
        llm_call_counter = 0
        print(f"🔄 Starting new compose operation (counter reset to 0)\n")

        initial_state = {
            "request": request,
            "draft": "",
            "critique": "",
            "revision_count": 0,
            "max_revisions": self.max_revisions,
            "final_email": ""
        }

        result = self.graph.invoke(initial_state)
        return {
            "final_email": result['final_email'],
            "revisions_made": result['revision_count'],
            "final_critique": result['critique']
        }

    def compose_with_history(self, request: str):
        """
        Compose an email and return full reflection history.

        Args:
            request: User's request describing the email to compose

        Returns:
            List of all drafts and critiques
        """
        history = []
        initial_state = {
            "request": request,
            "draft": "",
            "critique": "",
            "revision_count": 0,
            "max_revisions": self.max_revisions,
            "final_email": ""
        }

        for step in self.graph.stream(initial_state):
            history.append(step)

        return history
