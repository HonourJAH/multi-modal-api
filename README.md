# Multi-Modal API

An API that accepts text and/or image input and routes each request to the appropriate model — built with FastAPI and Ollama. Text-only prompts go to a lightweight LLM; anything with an attached image goes to a vision-language model for tasks like visual question answering.

---

## How It Works

```
POST /generate  (text only)        →  routes to the TEXT model
POST /generate  (text + image)     →  routes to the VISION model
POST /generate  (image only)       →  routes to the VISION model, using a default prompt
GET  /health                       →  health check
```

---

## Table of Contents

- [Why Route Between Two Models?](#why-route-between-two-models)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Running Tests](#running-tests)
- [API Endpoints](#api-endpoints)
- [Request & Response Schemas](#request--response-schemas)
- [Example Usage](#example-usage)
- [Docker](#docker)

---

## Why Route Between Two Models?

A single model rarely handles both text-only reasoning and image understanding equally well — vision-language models are heavier and slower, and using one for every request (even plain text) wastes compute for no benefit. This API makes that routing decision automatically, based purely on whether an image was attached:

```
No image attached   → OLLAMA_TEXT_MODEL    (fast, lightweight, text-only)
Image attached       → OLLAMA_VISION_MODEL  (multi-modal, handles VQA-style prompts)
```

The routing logic lives in one place (`app/services/ollama_client.py`), so the decision is made once, consistently, and is fully unit-tested without needing a live model server.

---

## Model Details

Both models run locally via [Ollama](https://ollama.com) — no external API calls, no per-request cost.

| Role | Model | Notes |
|---|---|---|
| Text | `llama3.2` | Lightweight, fast on CPU |
| Vision | `moondream` | Small vision-language model (~1.6GB) — chosen over larger options like `llava` specifically for CPU-only performance |

> **Note:** `moondream` was chosen deliberately after testing `llava` (4.6GB) on CPU-only hardware and hitting heavy memory-swap slowdowns. If your machine has more headroom (GPU, or 16GB+ RAM), `llava` is a stronger vision model — just set `OLLAMA_VISION_MODEL=llava` and `ollama pull llava`.

---

## Project Structure

```
multi-modal-api/
├── .github/
│   └── workflows/
│       └── ci.yml                — GitHub Actions CI pipeline
├── app/
│   ├── __init__.py
│   ├── main.py                    — FastAPI app, /generate route, shared httpx client lifecycle
│   ├── schemas.py                 — Response schema
│   └── services/
│       ├── __init__.py
│       └── ollama_client.py       — generate() — routes to text or vision model
├── tests/
│   ├── conftest.py                — Shared fixtures (fake httpx responses/client)
│   ├── test_ollama_client.py
│   └── test_main.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) installed and running **on your host machine** (not containerized — see [Docker](#docker) for why)
- Docker and Docker Compose (optional, for running the API itself in a container)

---

## Getting Started

### 1. Install Ollama

**Linux / macOS:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**

Download and run the installer from [ollama.com/download](https://ollama.com/download).

Confirm it's running:

```bash
curl http://localhost:11434
```

### 2. Pull the required models

```bash
ollama pull llama3.2
ollama pull moondream
```

### 3. Clone the repository

```bash
git clone https://github.com/HonourJAH/multi-modal-api.git
cd multi-modal-api
```

### 4. Create and activate a virtual environment

```bash
python3 -m venv venv
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

> **Note:** PowerShell may block the activation script by default due to its execution policy. If you get an error, run this once (as the current user, not admin) and try again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Start the API server

```bash
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where the API reaches Ollama |
| `OLLAMA_TEXT_MODEL` | `llama3.2` | Model used for text-only prompts |
| `OLLAMA_VISION_MODEL` | `moondream` | Model used for image+text prompts |

> **Important:** In Docker, `OLLAMA_BASE_URL` must be set to `http://host.docker.internal:11434`, never `localhost` — Ollama runs on the host machine, not in a container, and a container's own `localhost` refers to itself, not the host. `docker-compose.yml` already sets this correctly.

---

## Running Tests

Ollama is never called for real — `httpx.AsyncClient` is injected into `generate()`, and tests pass a mock directly instead of hitting a live model server.

```bash
pip install pytest pytest-asyncio
pytest -v
```

---

## API Endpoints

| Method | Endpoint | Description | Status Code |
|---|---|---|---|
| `POST` | `/generate` | Classify/respond to text and/or image input | `200 OK` |
| `GET` | `/health` | Health check | `200 OK` |

---

## Request & Response Schemas

### `POST /generate`

Accepts `multipart/form-data`, not JSON — required for file uploads.

| Field | Type | Required | Default |
|---|---|---|---|
| `prompt` | text | No | `"Describe this image."` |
| `image` | file | No | — |

At least one of `prompt` or `image` should be meaningful — sending neither still succeeds, but sends the default prompt to the text model with no image context.

**Response:**

```json
{
  "response": "The capital of France is Paris.",
  "model_used": "llama3.2",
  "input_type": "text"
}
```

`input_type` is either `"text"` or `"image+text"`, reflecting which path was taken.

---

### `GET /health`

```json
{ "status": "healthy" }
```

---

## Example Usage

### Text only

```bash
curl -X POST http://localhost:8000/generate \
  -F "prompt=What is the capital of France?"
```

### Image + text (visual question answering)

```bash
curl -X POST http://localhost:8000/generate \
  -F "prompt=What's in this image?" \
  -F "image=@/path/to/photo.jpg;type=image/jpeg"
```

### Image only (uses the default prompt)

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@/path/to/photo.jpg;type=image/jpeg"
```

---

## Docker

### Why Ollama isn't in `docker-compose.yml`

Unlike this project's API, Ollama is treated as existing host infrastructure rather than something this project owns — it's already installed and running directly on your machine (Step 1 above), with models already pulled. `docker-compose.yml` only defines the `api` service; it reaches your host's Ollama via `host.docker.internal`, which requires:

1. Ollama bound to `0.0.0.0`, not just `127.0.0.1` — set via a systemd override **(Linux only** — Windows/Mac don't use systemd; on those platforms, set the `OLLAMA_HOST` environment variable through the Ollama app's settings or your OS's environment variable configuration instead):

   ```bash
   sudo systemctl edit ollama
   ```

   ```ini
   [Service]
   Environment="OLLAMA_HOST=0.0.0.0:11434"
   ```

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```

2. `extra_hosts: - "host.docker.internal:host-gateway"` in `docker-compose.yml` (already included) — required on Linux; Docker Desktop on Mac/Windows resolves this automatically.

### Run with Docker Compose

```bash
docker compose up --build
```

### Stop

```bash
docker compose down
```

### Build the image only

```bash
docker build -t multi-modal-api .
```

### A note on CI

The GitHub Actions pipeline runs the full mocked test suite and validates that the Docker image builds successfully — it does **not** run a live smoke test against real Ollama, since Ollama runs on this project's host machine and isn't available on a GitHub-hosted runner. End-to-end verification against real models happens locally, as shown in [Example Usage](#example-usage).

---

## License

MIT
