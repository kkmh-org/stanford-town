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

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
