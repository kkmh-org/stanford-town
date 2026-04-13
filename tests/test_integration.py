"""
Integration tests — verify cognitive modules work end-to-end with
real Persona data and LLM calls.

Usage:
    cd generative_agents
    .venv/bin/pytest tests/test_integration.py -v
"""
import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'reverie', 'backend_server'))


# ---------------------------------------------------------------------------
# T-INT.2  Retrieve module
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_new_retrieve_returns_nodes(self, persona_with_memories):
        from persona.cognitive_modules.retrieve import new_retrieve
        persona = persona_with_memories
        result = new_retrieve(persona, ["Valentine's Day party"], 10)
        assert "Valentine's Day party" in result
        nodes = result["Valentine's Day party"]
        assert isinstance(nodes, list)
        assert len(nodes) <= 10
        assert len(nodes) > 0

    def test_new_retrieve_updates_last_accessed(self, persona_with_memories):
        from persona.cognitive_modules.retrieve import new_retrieve
        persona = persona_with_memories
        result = new_retrieve(persona, ["cafe"], 5)
        for node in result["cafe"]:
            assert node.last_accessed == persona.scratch.curr_time


# ---------------------------------------------------------------------------
# T-INT.4  Reflect module
# ---------------------------------------------------------------------------

class TestReflect:
    def test_reflect_trigger_creates_thoughts(self, persona_with_memories):
        from persona.cognitive_modules.reflect import reflect
        persona = persona_with_memories
        persona.scratch.importance_trigger_curr = 0
        thoughts_before = len(persona.a_mem.seq_thought)
        reflect(persona)
        thoughts_after = len(persona.a_mem.seq_thought)
        assert thoughts_after > thoughts_before

    def test_reflect_resets_counter(self, persona_with_memories):
        from persona.cognitive_modules.reflect import reflect
        persona = persona_with_memories
        persona.scratch.importance_trigger_curr = 0
        reflect(persona)
        assert persona.scratch.importance_trigger_curr == persona.scratch.importance_trigger_max
