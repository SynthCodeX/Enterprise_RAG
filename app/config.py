import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # Chat model for planner/responder. (prompt-guard is a 512-token classifier, not a chat model.)
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_GUARD_MODEL = os.getenv("GROQ_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")
    GUARD_THRESHOLD = float(os.getenv("GUARD_THRESHOLD", "0.5"))

    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_PRIMARY_SLUG = os.getenv("PORTKEY_PRIMARY_SLUG")
    PORTKEY_FALLBACK_SLUG = os.getenv("PORTKEY_FALLBACK_SLUG")
    PORTKEY_PRIMARY_CONFIG_ID = os.getenv("PORTKEY_PRIMARY_CONFIG_ID")

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LOGFIRE_BASE_URL = os.getenv("LOGFIRE_BASE_URL")
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

settings = Settings()
