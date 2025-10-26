import os, pathlib
from typing import List, Dict

os.environ.setdefault("CREWAI_TRACE_DISABLED", "1")
from crewai import Agent, Task, Crew, LLM

# ----- LLM config -----
def _llm() -> LLM:
    model = "llama3.2:1b"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    return LLM(
        model=f"ollama/{model}",
        base_url=base_url,
        temperature=temperature,
    )

def _system_prompt() -> str:
    p = pathlib.Path(__file__).resolve().parents[1] / "models" / "system_prompt.txt"
    return p.read_text(encoding="utf-8").strip()

# ----- Agent -----
def build_agent() -> Agent:
    return Agent(
        role="Chatbot",
        goal="Provide concise, correct answers to the user's latest message using conversation history.",
        backstory=_system_prompt(),
        llm=_llm(),
        verbose=False,
        allow_delegation=False,
    )

# ----- Helpers -----
_ALLOWED_ROLES = {"user", "assistant", "system"}

def _format_history(messages: List[Dict[str, str]], max_chars: int = 8000) -> str:
    # Keep roles safe and trim content
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        if role not in _ALLOWED_ROLES:
            role = "user"
        content = (m.get("content") or "").strip()
        # guard length per message
        if len(content) > 1200:
            content = content[:1200] + " ..."
        lines.append(f"{role}: {content}")
    transcript = "\n".join(lines)
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]
    return transcript

def _last_user(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return ""

# ----- Run chat with history -----
def run_chat(messages: List[Dict[str, str]]) -> str:
    """
    messages: list of dicts like {"role": "user"|"assistant"|"system", "content": "..."}
    """
    if not isinstance(messages, list) or not messages:
        return "Send at least one user message."

    history_text = _format_history(messages)
    last_user_msg = _last_user(messages)
    if not last_user_msg:
        return "No user query found."

    agent = build_agent()

    task = Task(
        description=(
            "You are in a multi-turn chat.\n\n"
            "Conversation so far:\n"
            f"{history_text}\n\n"
            "Task: Answer the LAST user message clearly and concisely. "
            "Use prior turns when relevant. If the user asks for step-by-step code or commands, provide them. "
            "If you must refuse, state the reason briefly and offer a safe alternative."
        ),
        expected_output="A concise, helpful answer to the last user message.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result).strip()
