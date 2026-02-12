# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a monorepo containing multiple AI-focused projects, each in its own directory:

- **emailmanager/**: Gmail AI agent using LangGraph and Google Gemini with agentic patterns (ReAct, Reflection, multi-step workflows)
- **travelagent/**: Travel itinerary planning agent using Google Gemini
- **irs-refund/**: IRS refund prediction with a Python ML model and Java Spring Boot REST API
- **train_fin/**: Financial data training pipeline
- **scratch/**: Experimental scripts and prototypes

## Common Commands

### Email Manager (emailmanager/)
```bash
cd emailmanager
pip install -r requirements.txt
python main.py
```
Requires: `GEMINI_API_KEY` in `.env` file, `credentials.json` from Google Cloud Console (OAuth 2.0)

### Travel Agent (travelagent/)
```bash
cd travelagent
pip install -r requirements.txt
python travelagentv2.py
```
Requires: `GEMINI_API_KEY` environment variable

### IRS Refund ML Model (irs-refund/ml-predict/)
```bash
cd irs-refund/ml-predict
python generate_data.py      # Generate synthetic training data
python train_model.py        # Train and save model
python serve_model.py        # Start Flask API server
```

### IRS Refund Spring Boot API (irs-refund/refund/)
```bash
cd irs-refund/refund
./gradlew build              # Build the project
./gradlew test               # Run tests
./gradlew bootRun            # Start the Spring Boot server
```

### Train Financial Pipeline (train_fin/)
```bash
cd train_fin
python main.py               # Runs full pipeline: create_additional_data -> train_model -> inference
```

## Architecture Notes

### Email Manager Architecture
The emailmanager project demonstrates three agentic patterns:

1. **ReAct Pattern** (`react_agent.py`): Implements reason-act-observe loops for email operations. Uses LangChain tools defined in `gmail_tools.py`.

2. **Reflection Pattern** (`reflection_composer.py`): Self-critique loop for email composition - generates draft, critiques, revises until approved.

3. **Multi-Step Workflow** (`email_prioritizer.py`): LangGraph StateGraph workflow: fetch_emails -> analyze_importance -> rank_emails -> generate_summary

Key files:
- `gmail_auth.py`: OAuth 2.0 authentication with Gmail API
- `gmail_tools.py`: LangChain tools wrapping Gmail operations (search, read, send, reply, delete)
- `config.py`: Configuration management

### Travel Agent Architecture
Single-file agent (`travelagentv2.py`) using Google GenAI SDK:
- Maintains conversation history as `types.Content` list
- Uses system instructions for persona definition
- Implements iterative plan generation with revision loop

### IRS Refund Architecture
Two-component system:
- **Python ML service** (`ml-predict/`): Trains scikit-learn model on synthetic data, serves predictions via Flask
- **Java Spring Boot API** (`refund/`): REST controller that interfaces with the ML service
