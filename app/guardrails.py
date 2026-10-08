import logfire
from langchain_groq import ChatGroq

from app.config import settings

BLOCKED_RESPONSE = "I can't help with that request."


def get_guard_llm() -> ChatGroq:
    """Groq-hosted prompt-injection classifier; model name comes from settings.GROQ_GUARD_MODEL."""
    return ChatGroq(
        model=settings.GROQ_GUARD_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
    )


_guard_llm = get_guard_llm()


def guard(query: str) -> tuple[bool, str]:
    """
    Returns (blocked, response). Prompt Guard 2 replies with a numeric
    injection/jailbreak score in [0, 1]; scores >= GUARD_THRESHOLD are blocked.
    Fails open (allows the request) if the guard model errors.
    """
    try:
        score = float(_guard_llm.invoke(query).content.strip())
    except Exception as e:
        logfire.warning(f"Prompt guard unavailable, allowing request: {e}")
        return False, ""

    blocked = score >= settings.GUARD_THRESHOLD
    logfire.info("Prompt guard score", score=score, blocked=blocked, model=settings.GROQ_GUARD_MODEL)
    return (True, BLOCKED_RESPONSE) if blocked else (False, "")
