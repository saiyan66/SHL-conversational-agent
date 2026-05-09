
# SHL Conversational Assessment Agent

## Features
- FastAPI backend
- Stateless conversational API
- Recommendation endpoint
- Health endpoint
- Dataset-ready architecture
- SHL assessment recommendation flow
- Easy VS Code setup

## Run Project

### Create virtual environment
```bash
python -m venv venv
```

### Activate
Windows:
```bash
venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run API
```bash
uvicorn app.main:app --reload
```

## Swagger Docs
http://127.0.0.1:8000/docs

## Dataset
Add your dataset here:

data/shl_catalog.json

Example:
[
  {
    "name": "Java 8 (New)",
    "url": "https://www.shl.com",
    "description": "Java assessment",
    "test_type": "K"
  }
]
