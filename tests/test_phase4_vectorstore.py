"""
Phase 4 tests — verify VectorStore integration in AssociativeMemory
and retrieve.py.

Usage:
    cd generative_agents
    .venv/bin/pytest tests/test_phase4_vectorstore.py -v
"""
import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'reverie', 'backend_server'))


from persona.prompt_template.gpt_structure import get_embedding, _embeddings
from persona.memory_structures.associative_memory import AssociativeMemory, ConceptNode


def _make_embedding_pair(text):
    """Helper: generate (text, vector) tuple."""
    return (text, get_embedding(text))


class TestVectorStoreBasics:
    """T4.1 — VectorStore basic operations."""

    def test_add_event_creates_vectorstore(self, tmp_path):
        """After adding an event, vectorstore should be non-None."""
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "embeddings.json").write_text("{}")
        (mem_dir / "nodes.json").write_text("{}")
        (mem_dir / "kw_strength.json").write_text(
            '{"kw_strength_event": {}, "kw_strength_thought": {}}')

        a_mem = AssociativeMemory(str(mem_dir))

        emb_pair = _make_embedding_pair("Isabella is cooking dinner")
        a_mem.add_event(
            created=datetime.datetime(2023, 2, 13, 10, 0),
            expiration=None,
            s="Isabella Rodriguez", p="is", o="cooking dinner",
            description="Isabella Rodriguez is cooking dinner",
            keywords={"Isabella Rodriguez", "cooking dinner"},
            poignancy=3,
            embedding_pair=emb_pair,
            filling=[],
        )
        assert a_mem.vectorstore is not None

    def test_vectorstore_search_finds_added_entry(self, tmp_path):
        """Added text can be found via relevance_search."""
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "embeddings.json").write_text("{}")
        (mem_dir / "nodes.json").write_text("{}")
        (mem_dir / "kw_strength.json").write_text(
            '{"kw_strength_event": {}, "kw_strength_thought": {}}')

        a_mem = AssociativeMemory(str(mem_dir))

        texts = [
            ("Isabella is cooking dinner", "cooking dinner", 3),
            ("Klaus is writing a paper", "writing paper", 5),
            ("Stock market crashed", "stock crash", 7),
        ]
        for desc, obj, poig in texts:
            emb_pair = _make_embedding_pair(desc)
            a_mem.add_event(
                created=datetime.datetime(2023, 2, 13, 10, 0),
                expiration=None,
                s="test", p="is", o=obj,
                description=desc,
                keywords={"test", obj},
                poignancy=poig,
                embedding_pair=emb_pair,
                filling=[],
            )

        results = a_mem.relevance_search("cooking food in kitchen", k=3)
        assert results is not None
        assert len(results) > 0
        top_key = max(results, key=results.get)
        assert "cooking" in top_key.lower()

    def test_vectorstore_save_and_reload(self, tmp_path):
        """VectorStore persists across save/load cycles."""
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "embeddings.json").write_text("{}")
        (mem_dir / "nodes.json").write_text("{}")
        (mem_dir / "kw_strength.json").write_text(
            '{"kw_strength_event": {}, "kw_strength_thought": {}}')

        a_mem = AssociativeMemory(str(mem_dir))
        emb_pair = _make_embedding_pair("test entry for persistence")
        a_mem.add_event(
            created=datetime.datetime(2023, 2, 13, 10, 0),
            expiration=None,
            s="test", p="is", o="persistent",
            description="test entry for persistence",
            keywords={"test"},
            poignancy=5,
            embedding_pair=emb_pair,
            filling=[],
        )

        save_dir = tmp_path / "save"
        save_dir.mkdir()
        a_mem.save(str(save_dir))
        assert (save_dir / "faiss_index").exists()

        a_mem2 = AssociativeMemory(str(save_dir))
        assert a_mem2.vectorstore is not None
        results = a_mem2.relevance_search("persistence test", k=1)
        assert results is not None
        assert len(results) > 0


class TestRelevanceSearch:
    """T4.2 — VectorStore relevance vs brute-force consistency."""

    def test_relevance_ranking_correct(self, tmp_path):
        """More relevant text should score higher than unrelated text."""
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "embeddings.json").write_text("{}")
        (mem_dir / "nodes.json").write_text("{}")
        (mem_dir / "kw_strength.json").write_text(
            '{"kw_strength_event": {}, "kw_strength_thought": {}}')

        a_mem = AssociativeMemory(str(mem_dir))

        entries = [
            "Isabella is preparing food for the Valentine's Day party",
            "Isabella is decorating Hobbs Cafe with hearts",
            "Klaus is writing about gentrification",
            "The weather is sunny today",
        ]
        for desc in entries:
            emb_pair = _make_embedding_pair(desc)
            a_mem.add_event(
                created=datetime.datetime(2023, 2, 13, 10, 0),
                expiration=None,
                s="test", p="is", o="test",
                description=desc,
                keywords={"test"},
                poignancy=5,
                embedding_pair=emb_pair,
                filling=[],
            )

        results = a_mem.relevance_search("Valentine's Day party food", k=4)
        assert results is not None

        sorted_keys = sorted(results, key=results.get, reverse=True)
        assert "party" in sorted_keys[0].lower() or "food" in sorted_keys[0].lower(), (
            f"Top result should be party/food related, got: {sorted_keys[0]}"
        )

    def test_dual_write_preserves_dict(self, tmp_path):
        """Adding entries still populates the legacy embeddings dict."""
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        (mem_dir / "embeddings.json").write_text("{}")
        (mem_dir / "nodes.json").write_text("{}")
        (mem_dir / "kw_strength.json").write_text(
            '{"kw_strength_event": {}, "kw_strength_thought": {}}')

        a_mem = AssociativeMemory(str(mem_dir))
        text = "test dual write"
        emb_pair = _make_embedding_pair(text)
        a_mem.add_event(
            created=datetime.datetime(2023, 2, 13, 10, 0),
            expiration=None,
            s="test", p="is", o="test",
            description=text,
            keywords={"test"},
            poignancy=5,
            embedding_pair=emb_pair,
            filling=[],
        )

        assert text in a_mem.embeddings
        assert isinstance(a_mem.embeddings[text], list)
        assert a_mem.vectorstore is not None
