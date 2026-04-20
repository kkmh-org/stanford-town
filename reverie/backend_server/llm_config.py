"""
LLM configuration for Generative Agents.

Centralizes LLM and Embedding settings, pointing to the team's LiteLLM gateway.
Override any value via environment variables.
"""
import os

LITELLM_BASE_URL = os.environ.get(
    "LITELLM_BASE_URL", "https://litellm.quickcan.com"
)
LITELLM_API_KEY = os.environ.get(
    "LITELLM_API_KEY", "sk-9B87dSinUvCGDaSa8ibLVw"
)

LLM_MODEL = os.environ.get("LLM_MODEL", "doubao-seed-1.8")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Langfuse observability
LANGFUSE_SECRET_KEY = os.environ.get(
    "LANGFUSE_SECRET_KEY", "sk-lf-74b16bd3-2625-4bad-b6d9-720a51d95cff"
)
LANGFUSE_PUBLIC_KEY = os.environ.get(
    "LANGFUSE_PUBLIC_KEY", "pk-lf-4c6fb617-8b54-427c-8939-01b6d6c3b1fe"
)
LANGFUSE_HOST = os.environ.get(
    "LANGFUSE_HOST", "https://langfuse.quickcan.com"
)
