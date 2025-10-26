from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
from .agent import run_chat

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=8000)

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    answer: str

app = FastAPI(title="Minimal Chatbot API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        history = req.messages[-12:]
        ans = run_chat([m.dict() for m in history])
        return ChatResponse(answer=ans)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
