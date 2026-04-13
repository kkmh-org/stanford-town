"""
Phase 2 tests — verify migrated LangChain prompt functions produce
outputs with correct types and ranges.

Usage:
    cd generative_agents
    .venv/bin/pytest tests/test_phase2_prompts.py -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'reverie', 'backend_server'))


class FakeScratch:
    """Minimal Scratch stand-in for testing prompt functions."""
    def __init__(self):
        self.name = "Isabella Rodriguez"
        self.first_name = "Isabella"
        self.last_name = "Rodriguez"
        self.age = 34
        self.innate = "friendly, outgoing, hospitable"
        self.learned = ("Isabella Rodriguez is a cafe owner of Hobbs Cafe "
                       "who loves to make people feel welcome.")
        self.currently = ("Isabella Rodriguez is planning on having a "
                         "Valentine's Day party at Hobbs Cafe.")
        self.lifestyle = "Isabella Rodriguez goes to bed around 11pm, wakes up around 6am."
        self.daily_plan_req = "Isabella Rodriguez opens Hobbs Cafe at 8am everyday."
        self.curr_time = __import__('datetime').datetime(2023, 2, 13, 10, 0, 0)

    def get_str_iss(self):
        return (f"Name: {self.name}\nAge: {self.age}\n"
                f"Innate traits: {self.innate}\n"
                f"Learned traits: {self.learned}\n"
                f"Currently: {self.currently}\n"
                f"Lifestyle: {self.lifestyle}\n"
                f"Daily plan requirement: {self.daily_plan_req}\n"
                f"Current Date: {self.curr_time.strftime('%A %B %d')}")

    def get_str_lifestyle(self):
        return self.lifestyle

    def get_str_firstname(self):
        return self.first_name

    def get_str_curr_date_str(self):
        return self.curr_time.strftime("%A %B %d")


class FakePersona:
    """Minimal Persona stand-in for testing prompt functions."""
    def __init__(self):
        self.name = "Isabella Rodriguez"
        self.scratch = FakeScratch()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_event_poignancy_returns_int_in_range():
    from persona.prompt_template.chain_utils import run_gpt_prompt_event_poignancy_v2
    persona = FakePersona()
    result, meta = run_gpt_prompt_event_poignancy_v2(persona, "eating breakfast")
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert 1 <= result <= 10, f"Expected 1-10, got {result}"


def test_thought_poignancy_returns_int_in_range():
    from persona.prompt_template.chain_utils import run_gpt_prompt_thought_poignancy_v2
    persona = FakePersona()
    result, meta = run_gpt_prompt_thought_poignancy_v2(
        persona, "Isabella realizes she forgot to order supplies for the party"
    )
    assert isinstance(result, int)
    assert 1 <= result <= 10


def test_chat_poignancy_returns_int_in_range():
    from persona.prompt_template.chain_utils import run_gpt_prompt_chat_poignancy_v2
    persona = FakePersona()
    result, meta = run_gpt_prompt_chat_poignancy_v2(
        persona, "chatting about the weather with a customer"
    )
    assert isinstance(result, int)
    assert 1 <= result <= 10


def test_pronunciatio_returns_emoji():
    from persona.prompt_template.chain_utils import run_gpt_prompt_pronunciatio_v2
    persona = FakePersona()
    result, meta = run_gpt_prompt_pronunciatio_v2("sleeping", persona)
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result) <= 3


def test_event_triple_returns_3_tuple():
    from persona.prompt_template.chain_utils import run_gpt_prompt_event_triple_v2
    persona = FakePersona()
    result, meta = run_gpt_prompt_event_triple_v2("cooking dinner", persona)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "Isabella Rodriguez"


def test_poignancy_mundane_vs_dramatic():
    """Mundane events should score lower than dramatic events."""
    from persona.prompt_template.chain_utils import run_gpt_prompt_event_poignancy_v2
    persona = FakePersona()
    mundane, _ = run_gpt_prompt_event_poignancy_v2(persona, "brushing teeth")
    dramatic, _ = run_gpt_prompt_event_poignancy_v2(persona, "her cafe is on fire")
    assert dramatic > mundane, (
        f"Dramatic ({dramatic}) should score higher than mundane ({mundane})"
    )


# ---------------------------------------------------------------------------
# P1 planning functions
# ---------------------------------------------------------------------------

def test_wake_up_hour_returns_reasonable_int():
    from persona.prompt_template.chain_utils import run_gpt_prompt_wake_up_hour_v2
    persona = FakePersona()
    result, meta = run_gpt_prompt_wake_up_hour_v2(persona)
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert 4 <= result <= 10, f"Expected 4-10, got {result}"


def test_daily_plan_returns_non_empty_list():
    from persona.prompt_template.chain_utils import run_gpt_prompt_daily_plan_v2
    persona = FakePersona()
    result, meta = run_gpt_prompt_daily_plan_v2(persona, wake_up_hour=6)
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) >= 4, f"Expected at least 4 items, got {len(result)}"
    assert result[0].startswith("wake up"), f"First item should start with 'wake up', got: {result[0]}"


def test_daily_plan_items_are_strings():
    from persona.prompt_template.chain_utils import run_gpt_prompt_daily_plan_v2
    persona = FakePersona()
    result, _ = run_gpt_prompt_daily_plan_v2(persona, wake_up_hour=7)
    for item in result:
        assert isinstance(item, str)
        assert len(item) > 5, f"Plan item too short: '{item}'"


# ---------------------------------------------------------------------------
# Batch A: simple string-output functions
# ---------------------------------------------------------------------------

def test_summarize_conversation():
    from persona.prompt_template.chain_utils import run_gpt_prompt_summarize_conversation_v2
    persona = FakePersona()
    convo = [["Alice", "Hi there!"], ["Bob", "Hello, how are you?"]]
    result, _ = run_gpt_prompt_summarize_conversation_v2(persona, convo)
    assert isinstance(result, str)
    assert "conversing" in result

def test_keyword_to_thoughts():
    from persona.prompt_template.chain_utils import run_gpt_prompt_keyword_to_thoughts_v2
    persona = FakePersona()
    result, _ = run_gpt_prompt_keyword_to_thoughts_v2(
        persona, "party", "Isabella is planning a Valentine's Day party")
    assert isinstance(result, str)
    assert len(result) > 5

def test_generate_whisper_inner_thought():
    from persona.prompt_template.chain_utils import run_gpt_prompt_generate_whisper_inner_thought_v2
    persona = FakePersona()
    result, _ = run_gpt_prompt_generate_whisper_inner_thought_v2(
        persona, "there is a Valentine's Day party coming up")
    assert isinstance(result, str)
    assert len(result) > 5

def test_planning_thought_on_convo():
    from persona.prompt_template.chain_utils import run_gpt_prompt_planning_thought_on_convo_v2
    persona = FakePersona()
    result, _ = run_gpt_prompt_planning_thought_on_convo_v2(
        persona, "Alice: Should we order a cake?\nBob: Yes, chocolate!")
    assert isinstance(result, str)
    assert len(result) > 3

def test_memo_on_convo():
    from persona.prompt_template.chain_utils import run_gpt_prompt_memo_on_convo_v2
    persona = FakePersona()
    result, _ = run_gpt_prompt_memo_on_convo_v2(
        persona, "Alice: I heard Klaus is writing about gentrification.\nBob: That's interesting.")
    assert isinstance(result, str)
    assert len(result) > 3

def test_act_obj_desc():
    from persona.prompt_template.chain_utils import run_gpt_prompt_act_obj_desc_v2
    persona = FakePersona()
    result, _ = run_gpt_prompt_act_obj_desc_v2("stove", "cooking dinner", persona)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Batch B: medium complexity functions
# ---------------------------------------------------------------------------

def test_act_obj_event_triple():
    from persona.prompt_template.chain_utils import run_gpt_prompt_act_obj_event_triple_v2
    persona = FakePersona()
    result, _ = run_gpt_prompt_act_obj_event_triple_v2("stove", "being used for cooking", persona)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "stove"

def test_focal_pt():
    from persona.prompt_template.chain_utils import run_gpt_prompt_focal_pt_v2
    persona = FakePersona()
    result, _ = run_gpt_prompt_focal_pt_v2(
        persona, "Isabella opened Hobbs Cafe.\nIsabella served coffee.\nKlaus is studying.", 3)
    assert isinstance(result, list)
    assert len(result) >= 1

def test_extract_keywords():
    from persona.prompt_template.chain_utils import run_gpt_prompt_extract_keywords_v2
    persona = FakePersona()
    result, _ = run_gpt_prompt_extract_keywords_v2(
        persona, "Isabella is preparing food for the Valentine's Day party")
    assert isinstance(result, set)
    assert len(result) > 0

def test_convo_to_thoughts():
    from persona.prompt_template.chain_utils import run_gpt_prompt_convo_to_thoughts_v2
    persona = FakePersona()
    result, _ = run_gpt_prompt_convo_to_thoughts_v2(
        persona, "Isabella", "Klaus",
        "Isabella: Hi Klaus!\nKlaus: Hey Isabella!", "Klaus")
    assert isinstance(result, str)
    assert len(result) > 3
