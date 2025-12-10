# Gmail AI Agent - Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Get Your API Keys

### Gemini API Key
1. Visit https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

### Gmail API Credentials
1. Go to https://console.cloud.google.com/
2. Create a new project (or select existing)
3. Enable Gmail API:
   - Click "Enable APIs and Services"
   - Search "Gmail API"
   - Click Enable
4. Create credentials:
   - Click "Create Credentials" → "OAuth client ID"
   - Application type: "Desktop app"
   - Download JSON file
   - Rename to `credentials.json`
   - Move to `emailmanager/` folder

## Step 2: Setup Environment

```bash
cd emailmanager

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env
# Replace 'your_key_here' with your actual Gemini API key
```

## Step 3: Run the App

```bash
python main.py
```

On first run:
- Browser will open for Gmail authentication
- Sign in with your Google account
- Grant permissions
- Authentication token saved to `token.json`

## Step 4: Try It Out!

### Example 1: Get Important Emails
```
Enter command: important
```
This will analyze your unread emails and show the top 5 most important ones.

### Example 2: Chat with Agent
```
Enter command: chat
You: Search for emails from my boss
```

### Example 3: Compose with AI
```
Enter command: compose
Describe the email: Write a thank you email to my team for their hard work
```

## Troubleshooting

### "GEMINI_API_KEY not found"
- Ensure `.env` file exists in `emailmanager/` directory
- Check that `GEMINI_API_KEY=your_actual_key` is set correctly

### "credentials.json not found"
- Download OAuth credentials from Google Cloud Console
- Ensure file is named exactly `credentials.json`
- Place it in the `emailmanager/` directory

### Authentication Issues
- Delete `token.json` and re-authenticate
- Ensure Gmail API is enabled in Google Cloud Console
- Check that OAuth consent screen is configured

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Try upgrading pip: `pip install --upgrade pip`

## What's Happening Under the Hood?

When you run a command, here's what happens:

### "important" Command
1. Fetches unread emails from Gmail
2. AI analyzes each email for importance
3. Scores emails 1-10 based on urgency, sender, content
4. Ranks and returns top 5
5. Generates executive summary

### "chat" Command (ReAct Pattern)
1. You ask a question
2. Agent **reasons** about what to do
3. Agent **acts** by calling Gmail tools
4. Agent **observes** the results
5. Repeats 2-4 until task complete
6. Responds to you

### "compose" Command (Reflection Pattern)
1. AI generates initial draft
2. AI critiques its own draft
3. AI revises based on critique
4. Repeats 2-3 up to 2 times
5. Returns final email

## Next Steps

- Read the full [README.md](README.md) for architecture details
- Explore the code to understand agentic patterns
- Modify `config.py` to customize behavior
- Add your own tools in `gmail_tools.py`

Happy email managing! 📧🤖
