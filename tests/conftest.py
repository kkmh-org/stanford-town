"""
Shared fixtures for generative_agents tests.
"""
import sys
import os
import datetime
import random

backend_dir = os.path.join(os.path.dirname(__file__),
                           '..', 'reverie', 'backend_server')
sys.path.insert(0, os.path.abspath(backend_dir))

import pytest

_storage = os.path.join(os.path.dirname(__file__), '..',
                        'environment', 'frontend_server', 'storage')
_sim = os.path.abspath(os.path.join(_storage,
                                    'base_the_ville_isabella_maria_klaus'))


@pytest.fixture
def persona_isabella():
    from persona.persona import Persona
    folder = os.path.join(_sim, 'personas', 'Isabella Rodriguez')
    p = Persona("Isabella Rodriguez", folder)
    p.scratch.curr_time = datetime.datetime(2023, 2, 13, 8, 0, 0)
    p.scratch.curr_tile = (58, 39)
    return p


@pytest.fixture
def persona_klaus():
    from persona.persona import Persona
    folder = os.path.join(_sim, 'personas', 'Klaus Mueller')
    p = Persona("Klaus Mueller", folder)
    p.scratch.curr_time = datetime.datetime(2023, 2, 13, 8, 0, 0)
    p.scratch.curr_tile = (60, 40)
    return p


@pytest.fixture
def personas(persona_isabella, persona_klaus):
    return {
        "Isabella Rodriguez": persona_isabella,
        "Klaus Mueller": persona_klaus,
    }


@pytest.fixture
def persona_with_memories(persona_isabella):
    from persona.prompt_template.gpt_structure import get_embedding
    persona = persona_isabella
    test_events = [
        "Isabella opened Hobbs Cafe this morning",
        "Isabella served coffee to a customer",
        "Isabella noticed Klaus studying at a table",
        "Isabella overheard a conversation about gentrification",
        "Isabella is planning the Valentine's Day party menu",
        "Isabella ordered red roses for the cafe decoration",
        "Isabella updated the cafe's social media page",
        "Isabella restocked the coffee beans",
        "Isabella chatted with Maria about the party",
        "Isabella printed flyers for the Valentine's Day event",
        "Isabella cleaned the espresso machine",
        "Isabella called the bakery for a cake order",
        "Isabella arranged tables for the party setup",
        "Isabella welcomed a group of students into the cafe",
        "Isabella reviewed the cafe's monthly expenses",
    ]
    for desc in test_events:
        emb_pair = (desc, get_embedding(desc))
        persona.a_mem.add_event(
            created=persona.scratch.curr_time,
            expiration=None,
            s="Isabella Rodriguez", p="is",
            o=desc.split("Isabella ")[-1],
            description=desc,
            keywords={"Isabella Rodriguez"},
            poignancy=random.randint(2, 8),
            embedding_pair=emb_pair, filling=[],
        )
        persona.scratch.importance_trigger_curr -= random.randint(2, 8)
        persona.scratch.importance_ele_n += 1
    return persona
