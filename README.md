# S-MAP AI Server

FastAPI backend for S-MAP services.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

If you prefer editable package installation:

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn app.main:app --reload
```

Open:

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/api/v1/health

## Project Layout

```text
app/
  api/                 # Versioned API routers
  core/                # Settings and shared configuration
  schemas/             # Request/response models
  services/            # Business logic integration points
tests/                 # API tests
```

## Test

```bash
pytest
```
