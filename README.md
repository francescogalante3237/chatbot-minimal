# Minimal Chatbot API

A basic FastAPI + CrewAI + Ollama application exposing a simple `/chat` endpoint to interact with a local LLM.

## How to Run

### 1. Prerequisites
- Docker and Docker Compose installed  
- If there is a local Ollama service, it must be stopped (port 11434 must be free)

### 2. Start the containers
docker compose up --build

This will:
- Start the Ollama service  
- Automatically pull the `llama3.2:1b` model  
- Launch the FastAPI app on port **8000**

## Test the API

### Health check
curl http://localhost:8000/health

Expected output:
{"status":"ok"}

### Chat endpoint
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain PCA in one sentence"}'

Example output:
{"answer":"PCA reduces data dimensionality by projecting it onto principal components."}

## Notes 
- `OLLAMA_BASE_URL` defaults to `http://ollama:11434` inside Docker.  
- Each `/chat` call spawns a new CrewAI agent for simplicity.  
- Minimal setup designed for clarity and reproducibility.
- Sometimes it gives dummy answers like "Though: now I can give great answers"
  It should be possible to prevent it by adding controls and parsing outputs.
