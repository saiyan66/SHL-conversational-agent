# SHL Assessment Conversational Agent

A stateless conversational agent that recommends SHL assessments based on user prompts. Built with FastAPI, Gemini 1.5 Flash, and TF-IDF retrieval.

## Endpoints

- `GET /health` – health check
- `POST /chat` – send conversation history, get agent reply + recommendations

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API_KEY
uvicorn app.main:app --reload
