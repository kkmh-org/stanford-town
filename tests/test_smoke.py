"""
Smoke tests — verify LiteLLM gateway connectivity and basic LLM/Embedding
functionality via the refactored gpt_structure module.

Run first after Phase 1 refactoring.  All three tests must pass before
proceeding.

Usage:
    cd generative_agents
    .venv/bin/pytest tests/test_smoke.py -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'reverie', 'backend_server'))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from llm_config import LITELLM_BASE_URL, LITELLM_API_KEY, LLM_MODEL, EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# T1.1  Gateway reachability
# ---------------------------------------------------------------------------

def test_llm_gateway_reachable():
    """LiteLLM gateway returns a non-empty string for a trivial prompt."""
    llm = ChatOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        model=LLM_MODEL,
    )
    response = llm.invoke("Say 'hello' and nothing else.")
    assert response.content is not None
    assert len(response.content.strip()) > 0


def test_llm_follows_instruction():
    """LLM can follow a simple instruction (return a specific number)."""
    llm = ChatOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        model=LLM_MODEL,
    )
    response = llm.invoke(
        "Return ONLY the number 42, with no other text."
    )
    assert "42" in response.content


def test_embedding_gateway_reachable():
    """Embedding API returns a float vector of reasonable dimension."""
    embeddings = OpenAIEmbeddings(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        model=EMBEDDING_MODEL,
    )
    vector = embeddings.embed_query("test text for embedding")
    assert isinstance(vector, list)
    assert len(vector) > 100
    assert all(isinstance(x, float) for x in vector)


# ---------------------------------------------------------------------------
# T1.2  Refactored gpt_structure function signatures
# ---------------------------------------------------------------------------

def test_chatgpt_request_returns_string():
    """ChatGPT_request returns a plain str (backward compat)."""
    from persona.prompt_template.gpt_structure import ChatGPT_request
    result = ChatGPT_request("Say hello.")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "ERROR" not in result


def test_gpt4_request_returns_string():
    """GPT4_request returns a plain str (backward compat)."""
    from persona.prompt_template.gpt_structure import GPT4_request
    result = GPT4_request("Say hello.")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "ERROR" not in result


def test_gpt_request_with_params():
    """GPT_request still accepts the legacy gpt_parameter dict."""
    from persona.prompt_template.gpt_structure import GPT_request
    gpt_param = {
        "engine": "text-davinci-003",
        "max_tokens": 50,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
    }
    result = GPT_request("Say hello.", gpt_param)
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_embedding_returns_float_list():
    """get_embedding returns a list of floats (backward compat)."""
    from persona.prompt_template.gpt_structure import get_embedding
    vector = get_embedding("hello world")
    assert isinstance(vector, list)
    assert len(vector) > 100
    assert all(isinstance(x, float) for x in vector)


def test_embedding_similarity_sanity():
    """Semantically similar texts have higher cosine sim than unrelated ones."""
    from persona.prompt_template.gpt_structure import get_embedding
    from numpy import dot
    from numpy.linalg import norm

    def cos_sim(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    emb_cat = get_embedding("The cat sat on the mat")
    emb_kitten = get_embedding("A kitten rested on the rug")
    emb_stock = get_embedding("Stock market crashed today")

    sim_related = cos_sim(emb_cat, emb_kitten)
    sim_unrelated = cos_sim(emb_cat, emb_stock)
    assert sim_related > sim_unrelated, (
        f"Related sim ({sim_related:.3f}) should be > "
        f"unrelated sim ({sim_unrelated:.3f})"
    )
