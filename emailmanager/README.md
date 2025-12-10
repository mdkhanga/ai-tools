# Gmail AI Agent - Agentic Email Management

An intelligent email management system powered by **LangGraph** and **Google Gemini**, demonstrating modern agentic AI patterns including ReAct, Reflection, and multi-step workflows.

## 🎯 Features

- **Intelligent Email Search** - Find emails using natural language or Gmail syntax
- **Email Reading & Management** - Read, reply, compose, send, and delete emails
- **AI-Powered Prioritization** - Automatically identify the 5 most important emails of the day
- **Smart Email Composition** - Draft emails with automatic reflection and improvement
- **Natural Language Interface** - Chat with the agent using conversational commands

## 🏗️ Architecture & Agentic Patterns

This project demonstrates three key agentic patterns:

### 1. ReAct (Reasoning + Acting) Pattern

**Location**: `react_agent.py`

The ReAct agent implements a thought-action-observation loop:
1. **Reason**: Analyze the user's request and decide what to do
2. **Act**: Execute Gmail tools to perform actions
3. **Observe**: Process results from tool execution
4. **Repeat**: Continue until task is complete

```
User Request → Reasoning → Tool Selection → Action → Observation → Reasoning → ...
```

**Example Flow**:
```
User: "Find unread emails from my boss and summarize them"
├─ Reason: Need to search for specific emails
├─ Act: Call search_emails tool with query "is:unread from:boss@company.com"
├─ Observe: Got 3 emails
├─ Reason: Need to read each email to summarize
├─ Act: Call read_email tool for each message
├─ Observe: Retrieved email contents
├─ Reason: Now I can summarize
└─ Respond: "Here's a summary of 3 unread emails from your boss..."
```

### 2. Reflection Pattern

**Location**: `reflection_composer.py`

The reflection pattern uses self-critique to improve outputs:
1. **Generate**: Create initial email draft
2. **Reflect**: Critically evaluate the draft
3. **Revise**: Improve based on critique
4. **Repeat**: Continue until approved or max iterations reached

```
Draft → Critique → Revise → Critique → Revise → Final Email
```

**LangGraph Workflow**:
```
generate_draft → reflect ⟷ revise
                    ↓
                finalize → END
```

**Example**:
```
Request: "Write a professional email declining a meeting"

Iteration 1:
  Draft: "I can't make it to the meeting."
  Critique: "Too casual, lacks professionalism and explanation"

Iteration 2:
  Draft: "Thank you for the invitation. Unfortunately, I have a scheduling
         conflict and won't be able to attend. I'd appreciate if we could
         reschedule or if I could receive the meeting notes."
  Critique: "APPROVED - Professional, courteous, provides alternative"
```

### 3. Multi-Step Workflow Pattern

**Location**: `email_prioritizer.py`

A structured LangGraph workflow for complex tasks:

```
fetch_emails → analyze_importance → rank_emails → generate_summary → END
```

**How It Works**:
1. **Fetch**: Retrieve unread emails from Gmail
2. **Analyze**: Use AI to score each email's importance (1-10)
3. **Rank**: Sort by importance score
4. **Summarize**: Generate executive summary of top emails

Each email is analyzed based on:
- Sender authority/importance
- Subject urgency
- Content requiring action
- Time sensitivity
- Business/personal relevance

## 📁 Project Structure

```
emailmanager/
├── config.py                   # Configuration management
├── gmail_auth.py              # Gmail OAuth authentication
├── gmail_tools.py             # LangChain tools for Gmail operations
├── react_agent.py             # ReAct agent implementation
├── reflection_composer.py     # Reflection pattern for email composition
├── email_prioritizer.py       # Multi-step workflow for prioritization
├── main.py                    # CLI interface
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Gmail account
- Google Cloud Console project with Gmail API enabled
- Gemini API key

### 2. Enable Gmail API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Gmail API:
   - Navigate to "APIs & Services" → "Library"
   - Search for "Gmail API"
   - Click "Enable"
4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"
   - Choose "Desktop app"
   - Download the credentials JSON file
   - Rename it to `credentials.json` and place it in the `emailmanager/` directory

### 3. Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Copy the key for the next step

### 4. Install Dependencies

```bash
cd emailmanager
pip install -r requirements.txt
```

### 5. Configure Environment

Create a `.env` file in the `emailmanager/` directory:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 6. Run the Application

```bash
python main.py
```

On first run, you'll be prompted to authenticate with Google. This will:
- Open a browser window
- Ask you to sign in to your Gmail account
- Request permission to manage your emails
- Save authentication token to `token.json` for future use

## 💻 Usage Examples

### Chat Mode (ReAct Agent)

```
Enter command: chat

You: Search for unread emails from john@example.com

Agent: [Searches emails and returns results]

You: Reply to the most recent one saying I'll review it tomorrow

Agent: [Crafts and sends reply]
```

### Important Emails (Multi-Step Workflow)

```
Enter command: important

✨ Analyzed 15 emails

TOP 5 MOST IMPORTANT EMAILS
==================================================
1. Q4 Budget Review Meeting
   From: ceo@company.com
   Importance: 9/10
   Why: High-priority meeting request from C-level executive

2. Production System Alert
   From: alerts@monitoring.com
   Importance: 8/10
   Why: Time-sensitive technical issue requiring immediate attention

[...]

EXECUTIVE SUMMARY
==================================================
Urgent items include C-level meeting request and production alert.
The client proposal needs response by EOD. Two action items require
your approval by end of week.
```

### Email Composition (Reflection Pattern)

```
Enter command: compose

Describe the email you want to compose:
> Write a professional follow-up email to a client about their pending proposal

🔄 Composing with reflection pattern...

FINAL EMAIL (after reflection)
==================================================
Subject: Follow-up on Your Proposal

Dear [Client Name],

I hope this email finds you well. I wanted to follow up on the
proposal you submitted last week regarding [project name].

We've reviewed the details carefully and are impressed with your
approach. We have a few questions that would help us move forward:

1. [Question 1]
2. [Question 2]

Could we schedule a brief call this week to discuss these points?
I'm available [provide availability].

Thank you for your patience, and I look forward to our continued
collaboration.

Best regards,
[Your Name]
==================================================
Revisions made: 1
Final assessment: APPROVED - Professional tone, clear purpose,
                  actionable next steps
```

## 🛠️ Available Gmail Tools

The agent has access to these tools (defined in `gmail_tools.py`):

- `search_emails(query, max_results)` - Search using Gmail syntax
- `read_email(email_id)` - Read full email content
- `compose_and_send_email(to, subject, body)` - Send new email
- `reply_to_email(email_id, reply_body)` - Reply to email
- `delete_email(email_id)` - Move email to trash
- `get_unread_emails(max_results)` - Get unread inbox emails

## 🧠 LangGraph Components

### State Management

Each workflow defines its state using TypedDict:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class EmailDraftState(TypedDict):
    request: str
    draft: str
    critique: str
    revision_count: int
    final_email: str
```

### Graph Construction

Workflows are built using StateGraph:

```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(gmail_tools))
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", should_continue)
graph = workflow.compile()
```

### Conditional Routing

Decisions are made using conditional edges:

```python
def should_continue(state):
    if has_tool_calls(state):
        return "continue"
    return "end"
```

## 🔒 Security & Privacy

- OAuth 2.0 authentication with Gmail
- Tokens stored locally in `token.json`
- No email data stored or transmitted to third parties
- All API calls are direct to Google and Gemini services
- Gemini API key stored in `.env` (not committed to git)

## 🎓 Learning Resources

To understand the patterns used:

- **LangGraph**: [Official Documentation](https://langchain-ai.github.io/langgraph/)
- **ReAct Pattern**: [Paper](https://arxiv.org/abs/2210.03629)
- **Reflection Pattern**: [LangChain Guide](https://langchain-ai.github.io/langgraph/tutorials/reflection/)
- **Gmail API**: [Google Documentation](https://developers.google.com/gmail/api)

## 🚧 Future Enhancements

- [ ] Add more sophisticated multi-agent collaboration
- [ ] Implement planning pattern for complex multi-email workflows
- [ ] Add email categorization and automatic labeling
- [ ] Implement scheduled email sending
- [ ] Add support for attachments
- [ ] Create web UI interface
- [ ] Add conversation memory across sessions
- [ ] Implement tool-calling with parallel execution

## 📝 License

MIT

## 🤝 Contributing

Contributions welcome! This is a demonstration project for learning agentic AI patterns.

---

**Built with**: LangGraph, LangChain, Google Gemini, Gmail API
