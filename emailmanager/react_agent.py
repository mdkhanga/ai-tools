from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from gmail_tools import gmail_tools
from config import Config
import operator


class AgentState(TypedDict):
    """State for the ReAct agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class EmailReActAgent:
    """
    ReAct (Reasoning + Acting) agent for email management.

    This agent uses the ReAct pattern:
    1. Reason: Analyze the user's request and decide what to do
    2. Act: Execute tools to perform actions
    3. Observe: Process tool results
    4. Repeat until task is complete
    """

    def __init__(self):
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.7
        )

        # Bind tools to the model
        self.llm_with_tools = self.llm.bind_tools(gmail_tools)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the ReAct agent graph"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(gmail_tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )

        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _call_model(self, state: AgentState):
        """Agent reasoning step - decide what to do next"""
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState):
        """Decide whether to continue or end"""
        messages = state["messages"]
        last_message = messages[-1]

        # If there are tool calls, continue
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        # Otherwise, end
        return "end"

    def run(self, user_input: str):
        """
        Run the ReAct agent with user input.

        Args:
            user_input: User's request/query

        Returns:
            Agent's response
        """
        # Create initial state with system message and user input
        system_message = """You are an intelligent email management assistant powered by AI.

You have access to Gmail tools that allow you to:
- Search for emails using Gmail search syntax
- Read email contents
- Compose and send new emails
- Reply to existing emails
- Delete emails (move to trash)
- Get unread emails

When the user asks you to perform email tasks, use the ReAct pattern:
1. REASON about what needs to be done
2. ACT by calling the appropriate tools
3. OBSERVE the results
4. Continue reasoning and acting until the task is complete

Be helpful, concise, and always confirm actions before sending emails on behalf of the user.
"""

        initial_state = {
            "messages": [
                HumanMessage(content=system_message),
                HumanMessage(content=user_input)
            ]
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Return the last message
        return result["messages"][-1].content

    def stream(self, user_input: str):
        """
        Stream the agent's execution step by step.

        Args:
            user_input: User's request/query

        Yields:
            Each step of the agent's execution
        """
        system_message = """You are an intelligent email management assistant powered by AI.

You have access to Gmail tools. Use the ReAct pattern to help users manage their emails effectively."""

        initial_state = {
            "messages": [
                HumanMessage(content=system_message),
                HumanMessage(content=user_input)
            ]
        }

        for step in self.graph.stream(initial_state):
            yield step
