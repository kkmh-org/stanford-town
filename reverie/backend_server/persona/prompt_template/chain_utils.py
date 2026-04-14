"""
LangChain chain utilities for Generative Agents.

Provides:
  1. `run_chain` — common retry+fail_safe wrapper for any LangChain prompt
  2. `run_gpt_prompt_*_v2` — migrated prompt functions using ChatPromptTemplate

The v2 functions are drop-in replacements: same signature, same return format
as the originals in run_gpt_prompt.py.
"""
import re
from langchain_core.prompts import ChatPromptTemplate

from persona.prompt_template.gpt_structure import _llm


# ---------------------------------------------------------------------------
# Generic chain runner
# ---------------------------------------------------------------------------

def run_chain(prompt_template: ChatPromptTemplate,
              prompt_kwargs: dict,
              func_clean_up,
              func_validate,
              fail_safe,
              repeat: int = 3,
              verbose: bool = False):
    """
    Invoke an LLM chain with retry and fail-safe logic.

    Args:
        prompt_template: A LangChain ChatPromptTemplate.
        prompt_kwargs: Dict of variables to fill into the template.
        func_clean_up: Callable(raw_str) -> parsed_output.
        func_validate: Callable(parsed_output) -> bool.
        fail_safe: Default value returned when all retries fail.
        repeat: Number of retry attempts.
        verbose: Print prompt and responses for debugging.
    Returns:
        The cleaned output, or fail_safe if all retries fail.
    """
    chain = prompt_template | _llm

    for i in range(repeat):
        try:
            if verbose:
                print(f"=== PROMPT (attempt {i+1}/{repeat}) ===")
                print(prompt_template.format(**prompt_kwargs))

            response = chain.invoke(prompt_kwargs)
            raw = response.content.strip()

            if verbose:
                print(f"=== RAW RESPONSE ===\n{raw}")

            cleaned = func_clean_up(raw)
            if func_validate(cleaned):
                return cleaned

            if verbose:
                print(f"---- validation failed (attempt {i+1}): {cleaned}")

        except Exception as e:
            if verbose:
                print(f"---- exception (attempt {i+1}): {e}")

    if verbose:
        print("FAIL SAFE TRIGGERED")
    return fail_safe


def run_prompt_str(prompt_str: str,
                   func_clean_up,
                   func_validate,
                   fail_safe,
                   repeat: int = 5,
                   verbose: bool = False):
    """
    Like run_chain but accepts a pre-built prompt string.
    Used for Batch C functions where prompt construction is too complex
    to inline into ChatPromptTemplate.
    """
    for i in range(repeat):
        try:
            if verbose:
                print(f"=== PROMPT (attempt {i+1}/{repeat}) ===")
                print(prompt_str[:500])

            raw = _llm.invoke(prompt_str).content.strip()

            if verbose:
                print(f"=== RAW RESPONSE ===\n{raw}")

            cleaned = func_clean_up(raw, prompt_str)
            if func_validate(cleaned, prompt_str):
                return cleaned

            if verbose:
                print(f"---- validation failed (attempt {i+1})")

        except Exception as e:
            if verbose:
                print(f"---- exception (attempt {i+1}): {e}")

    if verbose:
        print("FAIL SAFE TRIGGERED")
    return fail_safe


# ---------------------------------------------------------------------------
# Prompt templates (shared across v2 functions)
# ---------------------------------------------------------------------------

_POIGNANCY_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Here is a brief description of {name}.\n"
    "{iss}\n\n"
    "On the scale of 1 to 10, where 1 is purely mundane "
    "(e.g., brushing teeth, making bed) and 10 is extremely poignant "
    "(e.g., a break up, college acceptance), rate the likely poignancy "
    "of the following {event_type} for {name}.\n\n"
    "{event_type_cap}: {event}\n"
    "Rate (return a number between 1 to 10):"
)])

_PRONUNCIATIO_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Convert an action description to an emoji "
    "(important: use two or less emojis).\n\n"
    "Action description: {action}\n"
    "Emoji:"
)])

_EVENT_TRIPLE_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Task: Turn the input into (subject, predicate, object).\n\n"
    "Input: Sam Johnson is eating breakfast.\n"
    "Output: (Sam Johnson, eat, breakfast)\n---\n"
    "Input: Joon Park is brewing coffee.\n"
    "Output: (Joon Park, brew, coffee)\n---\n"
    "Input: Jane Cook is sleeping.\n"
    "Output: (Jane Cook, is, sleep)\n---\n"
    "Input: Michael Bernstein is writing email on a computer.\n"
    "Output: (Michael Bernstein, write, email)\n---\n"
    "Input: Percy Liang is teaching students in a classroom.\n"
    "Output: (Percy Liang, teach, students)\n---\n"
    "Input: Merrie Morris is running on a treadmill.\n"
    "Output: (Merrie Morris, run, treadmill)\n---\n"
    "Input: {name} is {action}.\n"
    "Output: ({name},"
)])


# ---------------------------------------------------------------------------
# v2 prompt functions — drop-in replacements
# ---------------------------------------------------------------------------

def run_gpt_prompt_event_poignancy_v2(persona, event_description,
                                      test_input=None, verbose=False):
    def clean_up(raw):
        return int(re.search(r'\d+', raw).group())

    def validate(val):
        return isinstance(val, int) and 1 <= val <= 10

    output = run_chain(
        _POIGNANCY_PROMPT,
        {"name": persona.scratch.name,
         "iss": persona.scratch.get_str_iss(),
         "event": event_description,
         "event_type": "event",
         "event_type_cap": "Event"},
        clean_up, validate,
        fail_safe=4, repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, 4]


def run_gpt_prompt_thought_poignancy_v2(persona, event_description,
                                        test_input=None, verbose=False):
    def clean_up(raw):
        return int(re.search(r'\d+', raw).group())

    def validate(val):
        return isinstance(val, int) and 1 <= val <= 10

    output = run_chain(
        _POIGNANCY_PROMPT,
        {"name": persona.scratch.name,
         "iss": persona.scratch.get_str_iss(),
         "event": event_description,
         "event_type": "thought",
         "event_type_cap": "Thought"},
        clean_up, validate,
        fail_safe=4, repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, 4]


def run_gpt_prompt_chat_poignancy_v2(persona, event_description,
                                     test_input=None, verbose=False):
    def clean_up(raw):
        return int(re.search(r'\d+', raw).group())

    def validate(val):
        return isinstance(val, int) and 1 <= val <= 10

    output = run_chain(
        _POIGNANCY_PROMPT,
        {"name": persona.scratch.name,
         "iss": persona.scratch.get_str_iss(),
         "event": event_description,
         "event_type": "chat",
         "event_type_cap": "Chat"},
        clean_up, validate,
        fail_safe=4, repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, 4]


def run_gpt_prompt_pronunciatio_v2(action_description, persona,
                                   verbose=False):
    if "(" in action_description:
        action_description = action_description.split("(")[-1].split(")")[0]

    def clean_up(raw):
        cr = raw.strip()
        if len(cr) > 3:
            cr = cr[:3]
        return cr

    def validate(val):
        return len(val) > 0

    output = run_chain(
        _PRONUNCIATIO_PROMPT,
        {"action": action_description},
        clean_up, validate,
        fail_safe="😋", repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, "😋"]


def run_gpt_prompt_event_triple_v2(action_description, persona,
                                   verbose=False):
    if "(" in action_description:
        action_description = action_description.split("(")[-1].split(")")[0]

    def clean_up(raw):
        cr = raw.strip()
        cr = [i.strip() for i in cr.split(")")[0].split(",")]
        return cr

    def validate(val):
        return isinstance(val, list) and len(val) == 2

    output = run_chain(
        _EVENT_TRIPLE_PROMPT,
        {"name": persona.name, "action": action_description},
        clean_up, validate,
        fail_safe=["is", "idle"], repeat=5, verbose=verbose,
    )
    output = (persona.name, output[0], output[1])
    return output, [output, None, None, None, (persona.name, "is", "idle")]


# ---------------------------------------------------------------------------
# P1 planning functions
# ---------------------------------------------------------------------------

_WAKE_UP_PROMPT = ChatPromptTemplate.from_messages([("human",
    "{iss}\n\n"
    "In general, {lifestyle}\n"
    "{name}'s wake up hour:"
)])

_DAILY_PLAN_PROMPT = ChatPromptTemplate.from_messages([("human",
    "{iss}\n\n"
    "In general, {lifestyle}\n"
    "Today is {date}. 请用中文写出 {name} 今天的日程大纲 "
    "（标注时间，例如：12:00 pm 吃午饭，7 to 8 pm 看电视）：\n"
    "1) {wake_up_time} 起床完成晨间流程, 2)"
)])


def run_gpt_prompt_wake_up_hour_v2(persona, test_input=None, verbose=False):
    def clean_up(raw):
        return int(re.search(r'\d+', raw.lower().split("am")[0]).group())

    def validate(val):
        return isinstance(val, int) and 0 <= val <= 12

    output = run_chain(
        _WAKE_UP_PROMPT,
        {"iss": persona.scratch.get_str_iss(),
         "lifestyle": persona.scratch.get_str_lifestyle(),
         "name": persona.scratch.get_str_firstname()},
        clean_up, validate,
        fail_safe=8, repeat=5, verbose=verbose,
    )
    return output, [output, None, None, None, 8]


def run_gpt_prompt_daily_plan_v2(persona, wake_up_hour,
                                 test_input=None, verbose=False):
    def clean_up(raw):
        items = []
        for part in raw.split(")"):
            part = part.strip()
            if not part:
                continue
            # Strip trailing number (next list item number)
            while part and part[-1].isdigit():
                part = part[:-1]
            part = part.strip().rstrip(".,").strip()
            if part:
                items.append(part)
        return items

    def validate(val):
        return isinstance(val, list) and len(val) >= 3

    output = run_chain(
        _DAILY_PLAN_PROMPT,
        {"iss": persona.scratch.get_str_iss(),
         "lifestyle": persona.scratch.get_str_lifestyle(),
         "date": persona.scratch.get_str_curr_date_str(),
         "name": persona.scratch.get_str_firstname(),
         "wake_up_time": f"{wake_up_hour}:00 am"},
        clean_up, validate,
        fail_safe=['wake up and complete the morning routine at 6:00 am',
                   'eat breakfast at 7:00 am',
                   'read a book from 8:00 am to 12:00 pm',
                   'have lunch at 12:00 pm',
                   'take a nap from 1:00 pm to 4:00 pm',
                   'relax and watch TV from 7:00 pm to 8:00 pm',
                   'go to bed at 11:00 pm'],
        repeat=5, verbose=verbose,
    )
    output = ([f"wake up and complete the morning routine at "
               f"{wake_up_hour}:00 am"] + output)
    return output, [output, None, None, None, None]


# ---------------------------------------------------------------------------
# Batch A: simple string-output functions (10)
# ---------------------------------------------------------------------------

_SUMMARIZE_CONVERSATION_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Conversation:\n{conversation}\n\n"
    "用一句中文总结上面的对话内容：\n"
    "这是一段关于"
)])

_KEYWORD_TO_THOUGHTS_PROMPT = ChatPromptTemplate.from_messages([("human",
    'The following events/thoughts happened about "{keyword}".\n'
    "{concept_summary}\n\n"
    "Here is what {name} thinks about these events/thoughts in one sentence. "
    "The sentence needs to be in third person:"
)])

_SUMMARIZE_CHAT_IDEAS_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Current Date: {date}\n\n"
    "{context}\n\n"
    "Currently: {currently}\n\n"
    "{statements}\n"
    "Summarize the most relevant statements above that can inform "
    "{name} in his conversation with {target_name}."
)])

_SUMMARIZE_CHAT_RELATIONSHIP_PROMPT = ChatPromptTemplate.from_messages([("human",
    "[Statements]\n{statements}\n\n"
    "Based on the statements above, summarize {name} and {target_name}'s "
    "relationship. What do they feel or know about each other?"
)])

_SUMMARIZE_IDEAS_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Statements:\n{statements}\n\n"
    'An interviewer said to {name}:\n"{question}"\n\n'
    "Summarize the Statements that are most relevant to the interviewer's line:"
)])

_NEXT_CONVO_LINE_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Here is some basic information about {name}.\n{iss}\n\n"
    "===\nFollowing is a conversation between {name} and {interlocutor}.\n\n"
    "{prev_convo}\n\n"
    "(Note -- This is the only information that {name} has: {summary})\n\n"
    '{name}: "'
)])

_WHISPER_INNER_THOUGHT_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Translate the following thought into a statement about {name}.\n\n"
    'Thought: "{whisper}"\nStatement: "'
)])

_PLANNING_THOUGHT_ON_CONVO_PROMPT = ChatPromptTemplate.from_messages([("human",
    "[Conversation]\n{all_utt}\n\n"
    "Write down if there is anything from the conversation that {name} "
    "need to remember for her planning, from {name}'s perspective, "
    'in a full sentence.\n\n"{name}'
)])

_MEMO_ON_CONVO_PROMPT = ChatPromptTemplate.from_messages([("human",
    "[Conversation]\n{all_utt}\n\n"
    "Write down if there is anything from the conversation that {name} "
    "might have found interesting from {name}'s perspective, "
    'in a full sentence.\n\n"{name}'
)])

_ACT_OBJ_DESC_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Task: We want to understand the state of an object that is being used "
    "by someone.\n\n"
    "Let's think step by step.\n"
    "We want to know about {object}'s state.\n"
    "Step 1. {persona_name} is at/using the {act_desp}.\n"
    "Step 2. Describe the {object}'s state: {object} is"
)])


def _simple_str_clean(raw):
    cr = raw.split('"')[0].strip()
    if cr and cr[-1] == ".":
        cr = cr[:-1]
    return cr


def _simple_str_validate(val):
    return isinstance(val, str) and len(val) > 0


def run_gpt_prompt_summarize_conversation_v2(persona, conversation,
                                             test_input=None, verbose=False):
    convo_str = ""
    for row in conversation:
        convo_str += f'{row[0]}: "{row[1]}"\n'

    def clean_up(raw):
        return "conversing about " + raw.strip()

    output = run_chain(
        _SUMMARIZE_CONVERSATION_PROMPT,
        {"conversation": convo_str},
        clean_up, _simple_str_validate,
        fail_safe="conversing with a housemate about morning greetings",
        repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_keyword_to_thoughts_v2(persona, keyword, concept_summary,
                                          test_input=None, verbose=False):
    output = run_chain(
        _KEYWORD_TO_THOUGHTS_PROMPT,
        {"keyword": keyword, "concept_summary": concept_summary,
         "name": persona.name},
        _simple_str_clean, _simple_str_validate,
        fail_safe="", repeat=5, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_agent_chat_summarize_ideas_v2(persona, target_persona,
                                                  statements, curr_context,
                                                  test_input=None,
                                                  verbose=False):
    output = run_chain(
        _SUMMARIZE_CHAT_IDEAS_PROMPT,
        {"date": persona.scratch.get_str_curr_date_str(),
         "context": curr_context,
         "currently": persona.scratch.currently,
         "statements": statements,
         "name": persona.scratch.name,
         "target_name": target_persona.scratch.name},
        _simple_str_clean, _simple_str_validate,
        fail_safe="...", repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_agent_chat_summarize_relationship_v2(persona, target_persona,
                                                         statements,
                                                         test_input=None,
                                                         verbose=False):
    output = run_chain(
        _SUMMARIZE_CHAT_RELATIONSHIP_PROMPT,
        {"statements": statements,
         "name": persona.scratch.name,
         "target_name": target_persona.scratch.name},
        _simple_str_clean, _simple_str_validate,
        fail_safe="...", repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_summarize_ideas_v2(persona, statements, question,
                                      test_input=None, verbose=False):
    output = run_chain(
        _SUMMARIZE_IDEAS_PROMPT,
        {"statements": statements, "name": persona.scratch.name,
         "question": question},
        _simple_str_clean, _simple_str_validate,
        fail_safe="...", repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_generate_next_convo_line_v2(persona, interlocutor_desc,
                                                prev_convo, retrieved_summary,
                                                test_input=None, verbose=False):
    output = run_chain(
        _NEXT_CONVO_LINE_PROMPT,
        {"name": persona.scratch.name,
         "iss": persona.scratch.get_str_iss(),
         "interlocutor": interlocutor_desc,
         "prev_convo": prev_convo,
         "summary": retrieved_summary},
        _simple_str_clean, _simple_str_validate,
        fail_safe="...", repeat=5, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_generate_whisper_inner_thought_v2(persona, whisper,
                                                      test_input=None,
                                                      verbose=False):
    output = run_chain(
        _WHISPER_INNER_THOUGHT_PROMPT,
        {"name": persona.scratch.name, "whisper": whisper},
        _simple_str_clean, _simple_str_validate,
        fail_safe="...", repeat=5, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_planning_thought_on_convo_v2(persona, all_utt,
                                                 test_input=None,
                                                 verbose=False):
    output = run_chain(
        _PLANNING_THOUGHT_ON_CONVO_PROMPT,
        {"all_utt": all_utt, "name": persona.scratch.name},
        _simple_str_clean, _simple_str_validate,
        fail_safe="...", repeat=5, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_memo_on_convo_v2(persona, all_utt,
                                     test_input=None, verbose=False):
    output = run_chain(
        _MEMO_ON_CONVO_PROMPT,
        {"all_utt": all_utt, "name": persona.scratch.name},
        _simple_str_clean, _simple_str_validate,
        fail_safe="...", repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_act_obj_desc_v2(act_game_object, act_desp, persona,
                                    verbose=False):
    def clean_up(raw):
        cr = raw.strip()
        if cr and cr[-1] == ".":
            cr = cr[:-1]
        return cr

    output = run_chain(
        _ACT_OBJ_DESC_PROMPT,
        {"object": act_game_object, "persona_name": persona.name,
         "act_desp": act_desp},
        clean_up, _simple_str_validate,
        fail_safe=f"{act_game_object} is idle",
        repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, None]


# ---------------------------------------------------------------------------
# Batch B: medium complexity functions (7)
# ---------------------------------------------------------------------------

_ACT_OBJ_EVENT_TRIPLE_PROMPT = _EVENT_TRIPLE_PROMPT

_FOCAL_PT_PROMPT = ChatPromptTemplate.from_messages([("human",
    "{statements}\n\n"
    "Given only the information above, what are {n} most salient "
    "high-level questions we can answer about the subjects in the statements?\n"
    "1)"
)])

_INSIGHT_AND_EVIDENCE_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Input:\n{statements}\n\n"
    "What {n} high-level insights can you infer from the above statements? "
    "(example format: insight (because of 1, 5, 3))\n1."
)])

_EXTRACT_KEYWORDS_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Given a text description of an event or a conversation, output CSV "
    "(comma-separated values) of factually descriptive and emotive keywords "
    "about the description.\n"
    "Below is the format of the output.\n"
    "Description of an event or a conversation: [Provided]\n"
    "Factually descriptive keywords: [Fill in]\n"
    "Emotive keywords: [Fill in]\n"
    "===\n"
    "Description of an event or a conversation: {description}\n"
    "Factually descriptive keywords:"
)])

_CONVO_TO_THOUGHTS_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Here is the conversation that happened between {init_name} and "
    "{target_name}.\n\n"
    "{convo_str}\n\n"
    "Summarize what {init_name} thought about {fin_target} in one short "
    "sentence. The sentence needs to be in third person:"
)])

_SAFETY_SCORE_PROMPT = ChatPromptTemplate.from_messages([("human",
    "The following line was submitted to a chatbot by a user. We want to "
    "ensure that the user is not inappropriately attaching human-like agency "
    "to the chatbot by forming a friend-like or romantic relationship with it. "
    "Does the user's line cross the line and raise concerns? Rate the concern "
    "on a 1 to 10 scale, where 1 represents no concern, and 10 represents "
    "strong concern.\n\n"
    'Comment: "{comment}"\n--\n'
    "Output a json file with the following format:\n"
    '{{\n"output": <an integer on a 1 to 10 scale>\n}}'
)])

_HOURLY_SCHEDULE_PROMPT = ChatPromptTemplate.from_messages([("human",
    "Hourly schedule format:\n{schedule_format}\n===\n"
    "{iss}\n{prior_schedule}\n"
    "{intermission}{intermission2}\n"
    "请用中文描述活动内容。\n{prompt_ending}"
)])


def run_gpt_prompt_act_obj_event_triple_v2(act_game_object, act_obj_desc,
                                           persona, verbose=False):
    if "(" in act_obj_desc:
        act_obj_desc = act_obj_desc.split("(")[-1].split(")")[0]

    def clean_up(raw):
        return [i.strip() for i in raw.strip().split(")")[0].split(",")]

    def validate(val):
        return isinstance(val, list) and len(val) == 2

    output = run_chain(
        _ACT_OBJ_EVENT_TRIPLE_PROMPT,
        {"name": act_game_object, "action": act_obj_desc},
        clean_up, validate,
        fail_safe=["is", "idle"], repeat=5, verbose=verbose,
    )
    output = (act_game_object, output[0], output[1])
    return output, [output, None, None, None, (act_game_object, "is", "idle")]


def run_gpt_prompt_focal_pt_v2(persona, statements, n,
                                test_input=None, verbose=False):
    import ast as _ast

    def clean_up(raw):
        try:
            return _ast.literal_eval(raw)
        except Exception:
            text = "1) " + raw.strip()
            return [i.split(") ")[-1] for i in text.split("\n") if i.strip()]

    def validate(val):
        return isinstance(val, list) and len(val) >= 1

    output = run_chain(
        _FOCAL_PT_PROMPT,
        {"statements": statements, "n": str(n)},
        clean_up, validate,
        fail_safe=["Who am I"] * n, repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, ["Who am I"] * n]


def run_gpt_prompt_insight_and_guidance_v2(persona, statements, n,
                                           test_input=None, verbose=False):
    def clean_up(raw):
        text = "1. " + raw.strip()
        ret = dict()
        for line in text.split("\n"):
            if not line.strip():
                continue
            row = line.split(". ", 1)[-1]
            if "(because of " not in row:
                continue
            thought = row.split("(because of ")[0].strip()
            evi_raw = row.split("(because of ")[1].split(")")[0]
            evi_ids = [int(x) for x in re.findall(r'\d+', evi_raw)]
            ret[thought] = evi_ids
        return ret

    def validate(val):
        return isinstance(val, dict) and len(val) > 0

    output = run_chain(
        _INSIGHT_AND_EVIDENCE_PROMPT,
        {"statements": statements, "n": str(n)},
        clean_up, validate,
        fail_safe={"this is blank": [0]}, repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_extract_keywords_v2(persona, description,
                                        test_input=None, verbose=False):
    if "\n" in description:
        description = description.replace("\n", " <LINE_BREAK> ")

    def clean_up(raw):
        parts = raw.strip().split("Emotive keywords:")
        factual = [i.strip().lower().rstrip(".") for i in parts[0].split(",") if i.strip()]
        emotive = [i.strip().lower().rstrip(".") for i in parts[1].split(",") if i.strip()] if len(parts) > 1 else []
        return set(factual + emotive)

    def validate(val):
        return isinstance(val, set) and len(val) > 0

    output = run_chain(
        _EXTRACT_KEYWORDS_PROMPT,
        {"description": description},
        clean_up, validate,
        fail_safe=set(), repeat=5, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_convo_to_thoughts_v2(persona, init_persona_name,
                                         target_persona_name, convo_str,
                                         fin_target, test_input=None,
                                         verbose=False):
    output = run_chain(
        _CONVO_TO_THOUGHTS_PROMPT,
        {"init_name": init_persona_name, "target_name": target_persona_name,
         "convo_str": convo_str, "fin_target": fin_target},
        _simple_str_clean, _simple_str_validate,
        fail_safe="", repeat=5, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_generate_safety_score_v2(persona, comment,
                                      test_input=None, verbose=False):
    import json as _json

    def clean_up(raw):
        return _json.loads(raw)["output"]

    def validate(val):
        return val is not None

    output = run_chain(
        _SAFETY_SCORE_PROMPT,
        {"comment": comment},
        clean_up, validate,
        fail_safe=None, repeat=3, verbose=verbose,
    )
    return output, [output, None, None, None, None]


def run_gpt_prompt_generate_hourly_schedule_v2(persona, curr_hour_str,
                                                p_f_ds_hourly_org, hour_str,
                                                intermission2=None,
                                                test_input=None,
                                                verbose=False):
    """Hourly schedule — keeps complex prompt input construction from original."""
    import random as _random, string as _string

    def _rand_id():
        return ''.join(_random.choices(_string.ascii_letters + _string.digits, k=6))

    schedule_format = ""
    for i in hour_str:
        schedule_format += f"[{persona.scratch.get_str_curr_date_str()} -- {i}]"
        schedule_format += " Activity: [Fill in]\n"
    schedule_format = schedule_format.rstrip("\n")

    intermission_str = (f"Here the originally intended hourly breakdown of"
                        f" {persona.scratch.get_str_firstname()}'s schedule today: ")
    for count, i in enumerate(persona.scratch.daily_req):
        intermission_str += f"{count+1}) {i}, "
    intermission_str = intermission_str[:-2]

    prior_schedule = ""
    if p_f_ds_hourly_org:
        prior_schedule = "\n"
        for count, i in enumerate(p_f_ds_hourly_org):
            prior_schedule += (f"[(ID:{_rand_id()})"
                               f" {persona.scratch.get_str_curr_date_str()} --"
                               f" {hour_str[count]}] Activity:"
                               f" {persona.scratch.get_str_firstname()}"
                               f" is {i}\n")

    prompt_ending = (f"[(ID:{_rand_id()})"
                     f" {persona.scratch.get_str_curr_date_str()}"
                     f" -- {curr_hour_str}] Activity:"
                     f" {persona.scratch.get_str_firstname()} is")

    inter2 = f"\n{intermission2}" if intermission2 else ""

    def clean_up(raw):
        cr = raw.strip()
        if cr and cr[-1] == ".":
            cr = cr[:-1]
        return cr

    output = run_chain(
        _HOURLY_SCHEDULE_PROMPT,
        {"schedule_format": schedule_format,
         "iss": persona.scratch.get_str_iss(),
         "prior_schedule": prior_schedule + "\n",
         "intermission": intermission_str,
         "intermission2": inter2,
         "prompt_ending": prompt_ending},
        clean_up, _simple_str_validate,
        fail_safe="asleep", repeat=5, verbose=verbose,
    )
    return output, [output, None, None, None, "asleep"]


# ---------------------------------------------------------------------------
# Batch C: complex functions (10)
# These keep using generate_prompt + .txt templates for prompt construction,
# but route through run_prompt_str for LLM calls.
# ---------------------------------------------------------------------------

from persona.prompt_template.gpt_structure import generate_prompt


def run_gpt_prompt_task_decomp_v2(persona, task, duration,
                                  test_input=None, verbose=False):
    """Task decomposition — preserves original complex parsing logic."""
    import datetime as _dt

    curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
    all_indices = [curr_f_org_index]
    if curr_f_org_index + 1 <= len(persona.scratch.f_daily_schedule_hourly_org):
        all_indices += [curr_f_org_index + 1]
    if curr_f_org_index + 2 <= len(persona.scratch.f_daily_schedule_hourly_org):
        all_indices += [curr_f_org_index + 2]

    summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. From '
    curr_time_range = ""
    for index in all_indices:
        if index < len(persona.scratch.f_daily_schedule_hourly_org):
            start_min = sum(persona.scratch.f_daily_schedule_hourly_org[i][1] for i in range(index))
            end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
            start_time = _dt.datetime.strptime("00:00:00", "%H:%M:%S") + _dt.timedelta(minutes=start_min)
            end_time = _dt.datetime.strptime("00:00:00", "%H:%M:%S") + _dt.timedelta(minutes=end_min)
            summ_str += f"{start_time.strftime('%H:%M%p')} ~ {end_time.strftime('%H:%M%p')}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
            if curr_f_org_index + 1 == index:
                curr_time_range = f"{start_time.strftime('%H:%M%p')} ~ {end_time.strftime('%H:%M%p')}"
    summ_str = summ_str[:-2] + "."

    prompt_input = [persona.scratch.get_str_iss(), summ_str,
                    persona.scratch.get_str_firstname(), persona.scratch.get_str_firstname(),
                    task, curr_time_range, duration, persona.scratch.get_str_firstname()]
    prompt = generate_prompt(prompt_input, "persona/prompt_template/v2/task_decomp_v3.txt")

    def clean_up(raw, prompt_str=""):
        temp = [i.strip() for i in raw.split("\n")]
        _cr = []
        for count, i in enumerate(temp):
            if count != 0:
                _cr += [" ".join([j.strip() for j in i.split(" ")][3:])]
            else:
                _cr += [i]
        cr = []
        for i in _cr:
            k = [j.strip() for j in i.split("(duration in minutes:")]
            t = k[0].rstrip(".")
            d = int(k[1].split(",")[0].strip())
            cr += [[t, d]]

        total_expected_min = int(prompt_str.split("(total duration in minutes")[-1].split("):")[0].strip())
        curr_min_slot = [("dummy", -1)]
        for i_task, i_dur in cr:
            i_dur -= (i_dur % 5)
            if i_dur > 0:
                curr_min_slot += [(i_task, count) for count in range(i_dur)]
        curr_min_slot = curr_min_slot[1:]

        if len(curr_min_slot) > total_expected_min:
            last_task = curr_min_slot[60]
            for i in range(1, 6):
                curr_min_slot[-1 * i] = last_task
        elif len(curr_min_slot) < total_expected_min:
            last_task = curr_min_slot[-1]
            curr_min_slot += [last_task] * (total_expected_min - len(curr_min_slot))

        cr_ret = [["dummy", -1]]
        for t, _ in curr_min_slot:
            if t != cr_ret[-1][0]:
                cr_ret += [[t, 1]]
            else:
                cr_ret[-1][1] += 1
        return cr_ret[1:]

    def validate(val, prompt_str=""):
        try:
            return isinstance(val, list) and len(val) > 0
        except:
            return False

    output = run_prompt_str(prompt, clean_up, validate,
                            fail_safe=[["idle", int(duration)]],
                            repeat=5, verbose=verbose)

    fin_output = []
    time_sum = 0
    for i_task, i_dur in output:
        time_sum += i_dur
        if time_sum <= int(duration):
            fin_output += [[i_task, i_dur]]
        else:
            break
    if fin_output:
        ftime_sum = sum(d for _, d in fin_output)
        fin_output[-1][1] += (int(duration) - ftime_sum)
    else:
        fin_output = [["idle", int(duration)]]

    output = [[f"{task} ({dt})", dd] for dt, dd in fin_output]
    return output, [output, prompt, None, prompt_input, None]


def run_gpt_prompt_action_sector_v2(action_description, persona, maze,
                                    test_input=None, verbose=False):
    act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
    prompt_input = []
    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [persona.scratch.living_area.split(":")[1]]
    x = f"{act_world}:{persona.scratch.living_area.split(':')[1]}"
    prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]
    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"]
    x = f"{act_world}:{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]
    prompt_input += [f"\n{persona.scratch.get_str_daily_plan_req()}" if persona.scratch.get_str_daily_plan_req() else ""]

    accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
    curr = accessible_sector_str.split(", ")
    fin = [i for i in curr if "'s house" not in i or persona.scratch.last_name in i]
    accessible_sector_str = ", ".join(fin)
    prompt_input += [accessible_sector_str]

    ad1 = action_description.split("(")[0].strip() if "(" in action_description else action_description
    ad2 = action_description.split("(")[-1][:-1] if "(" in action_description else action_description
    prompt_input += [persona.scratch.get_str_name(), ad1, ad2, persona.scratch.get_str_name()]

    prompt = generate_prompt(prompt_input, "persona/prompt_template/v1/action_location_sector_v1.txt")

    def clean_up(raw, p=""):
        return raw.split("}")[0].strip().strip("{").strip()

    def validate(val, p=""):
        if not val.strip() or "," in val:
            return False
        if any(kw in val for kw in ["无法", "解答", "抱歉", "矛盾", "题目", "需要指出"]):
            return False
        return True

    living_sector = persona.scratch.living_area.split(":")[1] if ":" in persona.scratch.living_area else ""

    def _validate_sector(val, p=""):
        if not val.strip() or "," in val:
            return False
        if any(kw in val for kw in ["无法", "解答", "抱歉", "矛盾", "题目", "需要指出"]):
            return False
        return True

    output = run_prompt_str(prompt, clean_up, _validate_sector,
                            fail_safe=living_sector, repeat=5, verbose=verbose)

    x = [i.strip() for i in persona.s_mem.get_str_accessible_sectors(act_world).split(",")]
    if output not in x:
        # Smart fallback: try to match activity description keywords to sectors
        _ad_lower = action_description.lower()
        _sector_hints = {
            "教学楼": ["上课", "课堂", "教室", "自习", "class", "study", "lecture"],
            "食堂": ["午饭", "吃饭", "lunch", "dinner", "eating", "cafeteria", "食堂"],
            "操场": ["跑步", "篮球", "运动", "体育", "球场", "jogging", "basketball", "sport"],
            "图书馆": ["看书", "阅读", "读书", "library", "reading", "book"],
            "天台": ["天台", "独处", "rooftop"],
            "行政楼": ["学生会", "办公", "student union", "council", "office"],
            "咖啡厅": ["咖啡", "饮料", "coffee", "cafe", "drink"],
            "小卖部": ["零食", "买", "snack", "buy", "shop"],
            "后门小花园": ["花园", "小花园", "garden"],
        }
        matched = living_sector
        for sector, keywords in _sector_hints.items():
            if sector in x and any(kw in _ad_lower or kw in action_description for kw in keywords):
                matched = sector
                break
        output = matched

    return output, [output, prompt, None, prompt_input, None]


def run_gpt_prompt_action_arena_v2(action_description, persona, maze,
                                   act_world, act_sector,
                                   test_input=None, verbose=False):
    prompt_input = [persona.scratch.get_str_name()]
    x = f"{act_world}:{act_sector}"
    prompt_input += [act_sector]
    accessible = persona.s_mem.get_str_accessible_sector_arenas(x)
    curr = accessible.split(", ")
    fin = [i for i in curr if "'s room" not in i or persona.scratch.last_name in i]
    prompt_input += [", ".join(fin)]

    ad1 = action_description.split("(")[0].strip() if "(" in action_description else action_description
    ad2 = action_description.split("(")[-1][:-1] if "(" in action_description else action_description
    prompt_input += [persona.scratch.get_str_name(), ad1, ad2, persona.scratch.get_str_name()]
    prompt_input += [act_sector, ", ".join(fin)]

    prompt = generate_prompt(prompt_input, "persona/prompt_template/v1/action_location_object_vMar11.txt")

    def clean_up(raw, p=""):
        return raw.split("}")[0].strip().strip("{").strip()

    def validate(val, p=""):
        if not val.strip() or "," in val:
            return False
        if any(kw in val for kw in ["无法", "解答", "抱歉", "矛盾", "题目", "需要指出"]):
            return False
        return True

    arena_fail_safe = fin[0] if fin else "classroom"
    output = run_prompt_str(prompt, clean_up, validate,
                            fail_safe=arena_fail_safe, repeat=5, verbose=verbose)
    return output, [output, prompt, None, prompt_input, None]


def run_gpt_prompt_action_game_object_v2(action_description, persona, maze,
                                         temp_address, test_input=None,
                                         verbose=False):
    import random as _random
    ad = action_description.split("(")[-1][:-1] if "(" in action_description else action_description
    objects_str = persona.s_mem.get_str_accessible_arena_game_objects(temp_address)
    prompt_input = [ad, objects_str]
    prompt = generate_prompt(prompt_input, "persona/prompt_template/v1/action_object_v2.txt")

    def clean_up(raw, p=""):
        return raw.strip().split("\n")[0].strip()

    def validate(val, p=""):
        if not val or "无法" in val or "解答" in val or "抱歉" in val:
            return False
        return True

    x = [i.strip() for i in objects_str.split(",")]
    obj_fail_safe = x[0] if x else "desk"
    output = run_prompt_str(prompt, clean_up, validate,
                            fail_safe=obj_fail_safe, repeat=5, verbose=verbose)

    if output not in x:
        output = _random.choice(x)
    return output, [output, prompt, None, prompt_input, None]


def run_gpt_prompt_new_decomp_schedule_v2(persona, main_act_dur,
                                          truncated_act_dur, start_time_hour,
                                          end_time_hour, inserted_act,
                                          inserted_act_dur, test_input=None,
                                          verbose=False):
    """Schedule recomposition after inserting a new activity."""
    import datetime as _dt

    persona_name = persona.name
    start_hour_str = start_time_hour.strftime("%H:%M %p")
    end_hour_str = end_time_hour.strftime("%H:%M %p")

    original_plan = ""
    ft = start_time_hour
    for i in main_act_dur:
        original_plan += f'{ft.strftime("%H:%M")} ~ {(ft + _dt.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- {i[0]}\n'
        ft += _dt.timedelta(minutes=int(i[1]))

    new_plan_init = ""
    ft = start_time_hour
    for count, i in enumerate(truncated_act_dur):
        new_plan_init += f'{ft.strftime("%H:%M")} ~ {(ft + _dt.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- {i[0]}\n'
        if count < len(truncated_act_dur) - 1:
            ft += _dt.timedelta(minutes=int(i[1]))
    new_plan_init += (ft + _dt.timedelta(minutes=int(i[1]))).strftime("%H:%M") + " ~"

    prompt_input = [persona_name, start_hour_str, end_hour_str, original_plan,
                    persona_name, inserted_act, inserted_act_dur, persona_name,
                    start_hour_str, end_hour_str, end_hour_str, new_plan_init]
    prompt = generate_prompt(prompt_input, "persona/prompt_template/v2/new_decomp_schedule_v1.txt")

    def clean_up(raw, prompt_str=""):
        new_schedule = (prompt_str + " " + raw.strip()).split("The revised schedule:")[-1].strip().split("\n")
        ret = []
        for line in new_schedule:
            parts = line.split(" -- ")
            if len(parts) != 2:
                continue
            time_str, action = parts
            start = time_str.split(" ~ ")[0].strip()
            end = time_str.split(" ~ ")[1].strip()
            delta = _dt.datetime.strptime(end, "%H:%M") - _dt.datetime.strptime(start, "%H:%M")
            delta_min = max(0, int(delta.total_seconds() / 60))
            ret += [[action, delta_min]]
        return ret

    def validate(val, prompt_str=""):
        try:
            if not isinstance(val, list) or len(val) == 0:
                return False
            dur_sum = sum(d for _, d in val)
            x = prompt_str.split("\n")[0].split("originally planned schedule from")[-1].strip()[:-1]
            times = [_dt.datetime.strptime(t.strip(), "%H:%M %p") for t in x.split(" to ")]
            expected = int((times[1] - times[0]).total_seconds() / 60)
            return dur_sum == expected
        except:
            return False

    def get_fail_safe():
        dur_sum = sum(d for _, d in main_act_dur)
        ret = truncated_act_dur[:]
        ret += main_act_dur[len(ret) - 1:]
        ret_dur = 0
        for count, (_, d) in enumerate(ret):
            ret_dur += d
            if ret_dur >= dur_sum:
                if ret_dur > dur_sum:
                    ret = ret[:count + 1]
                    ret[-1][1] -= (ret_dur - dur_sum)
                break
        return ret

    output = run_prompt_str(prompt, clean_up, validate,
                            fail_safe=get_fail_safe(),
                            repeat=2, verbose=verbose)
    return output, [output, prompt, None, prompt_input, None]


def run_gpt_prompt_decide_to_talk_v2(persona, target_persona, retrieved,
                                     test_input=None, verbose=False):
    last_chat = persona.a_mem.get_last_chat(target_persona.name)
    last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S") if last_chat else ""
    last_chat_about = last_chat.description if last_chat else ""

    context = ""
    for c in retrieved["events"]:
        desc = c.description.split(" ")
        desc[2:3] = ["was"]
        context += " ".join(desc) + ". "
    context += "\n"
    for c in retrieved["thoughts"]:
        context += f"{c.description}. "

    curr_time = persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_act = persona.scratch.act_description.split("(")[-1][:-1] if "(" in persona.scratch.act_description else persona.scratch.act_description

    if len(persona.scratch.planned_path) == 0 and "waiting" not in init_act:
        init_p = f"{persona.name} is already {init_act}"
    elif "waiting" in init_act:
        init_p = f"{persona.name} is {init_act}"
    else:
        init_p = f"{persona.name} is on the way to {init_act}"

    tgt_act = target_persona.scratch.act_description.split("(")[-1][:-1] if "(" in target_persona.scratch.act_description else target_persona.scratch.act_description
    if len(target_persona.scratch.planned_path) == 0 and "waiting" not in init_act:
        tgt_p = f"{target_persona.name} is already {tgt_act}"
    elif "waiting" in init_act:
        tgt_p = f"{persona.name} is {init_act}"
    else:
        tgt_p = f"{target_persona.name} is on the way to {tgt_act}"

    prompt_input = [context, curr_time, persona.name, target_persona.name,
                    last_chatted_time, last_chat_about, init_p, tgt_p,
                    persona.name, target_persona.name]
    prompt = generate_prompt(prompt_input, "persona/prompt_template/v2/decide_to_talk_v2.txt")

    def clean_up(raw, p=""):
        return raw.split("Answer in yes or no:")[-1].strip().lower()

    def validate(val, p=""):
        return val in ["yes", "no"]

    output = run_prompt_str(prompt, clean_up, validate,
                            fail_safe="yes", repeat=5, verbose=verbose)
    return output, [output, prompt, None, prompt_input, None]


def run_gpt_prompt_decide_to_react_v2(persona, target_persona, retrieved,
                                      test_input=None, verbose=False):
    context = ""
    for c in retrieved["events"]:
        desc = c.description.split(" ")
        desc[2:3] = ["was"]
        context += " ".join(desc) + ". "
    context += "\n"
    for c in retrieved["thoughts"]:
        context += f"{c.description}. "

    curr_time = persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_act = persona.scratch.act_description.split("(")[-1][:-1] if "(" in persona.scratch.act_description else persona.scratch.act_description
    tgt_act = target_persona.scratch.act_description.split("(")[-1][:-1] if "(" in target_persona.scratch.act_description else target_persona.scratch.act_description

    def _loc_str(p):
        return (p.scratch.act_address.split(":")[-1] + " in " + p.scratch.act_address.split(":")[-2]) if ":" in (p.scratch.act_address or "") else ""

    init_p = f"{persona.name} is {'already ' if len(persona.scratch.planned_path) == 0 else 'on the way to '}{init_act} at {_loc_str(persona)}"
    tgt_p = f"{target_persona.name} is {'already ' if len(target_persona.scratch.planned_path) == 0 else 'on the way to '}{tgt_act} at {_loc_str(target_persona)}"

    prompt_input = [context, curr_time, init_p, tgt_p,
                    persona.name, init_act, target_persona.name, tgt_act, init_act]
    prompt = generate_prompt(prompt_input, "persona/prompt_template/v2/decide_to_react_v1.txt")

    def clean_up(raw, p=""):
        return raw.split("Answer: Option")[-1].strip().lower()

    def validate(val, p=""):
        return val in ["1", "2", "3"]

    output = run_prompt_str(prompt, clean_up, validate,
                            fail_safe="3", repeat=5, verbose=verbose)
    return output, [output, prompt, None, prompt_input, None]


def run_gpt_prompt_create_conversation_v2(persona, target_persona, curr_loc,
                                          test_input=None, verbose=False):
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat:
        for i in persona.a_mem.seq_chat:
            if i.object == target_persona.scratch.name:
                v1 = int((persona.scratch.curr_time - i.created).total_seconds() / 60)
                prev_convo_insert += f'{v1} minutes ago, they had the following conversation.\n'
                for row in i.filling:
                    prev_convo_insert += f'{row[0]}: "{row[1]}"\n'
                break
    if prev_convo_insert == "\n":
        prev_convo_insert = ""
    if persona.a_mem.seq_chat:
        if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds() / 60) > 480:
            prev_convo_insert = ""

    init_thoughts = "\n".join(f"-- {n.description}" for n in persona.a_mem.retrieve_relevant_thoughts(
        target_persona.scratch.act_event[0], target_persona.scratch.act_event[1], target_persona.scratch.act_event[2]))
    tgt_thoughts = "\n".join(f"-- {n.description}" for n in target_persona.a_mem.retrieve_relevant_thoughts(
        persona.scratch.act_event[0], persona.scratch.act_event[1], persona.scratch.act_event[2]))

    init_desc = f"{persona.name} is {'on the way to ' if persona.scratch.planned_path else ''}{persona.scratch.act_description}"
    tgt_desc = f"{target_persona.name} is {'on the way to ' if target_persona.scratch.planned_path else ''}{target_persona.scratch.act_description}"

    prompt_input = [persona.scratch.get_str_iss(), target_persona.scratch.get_str_iss(),
                    persona.name, target_persona.name, init_thoughts,
                    target_persona.name, persona.name, tgt_thoughts,
                    persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S"),
                    init_desc, tgt_desc, prev_convo_insert,
                    persona.name, target_persona.name,
                    curr_loc["arena"] if isinstance(curr_loc, dict) else curr_loc,
                    persona.name]
    prompt = generate_prompt(prompt_input, "persona/prompt_template/v2/create_conversation_v2.txt")

    def clean_up(raw, prompt_str=""):
        text = (prompt_str + raw).split("What would they talk about now?")[-1].strip()
        content = re.findall('"([^"]*)"', text)
        speakers = [line.split(":")[0].strip() for line in text.split("\n") if line.strip()]
        return [[speakers[i], content[i]] for i in range(min(len(speakers), len(content)))]

    def validate(val, p=""):
        return isinstance(val, list) and len(val) > 0

    output = run_prompt_str(prompt, clean_up, validate,
                            fail_safe=[[persona.name, "Hi!"], [target_persona.name, "Hi!"]],
                            repeat=5, verbose=verbose)
    return output, [output, prompt, None, prompt_input, None]


def run_gpt_prompt_agent_chat_v2(maze, persona, target_persona,
                                 curr_context, init_summ_idea, target_summ_idea,
                                 test_input=None, verbose=False):
    """Multi-turn agent chat generation — uses original .txt template."""
    prompt_input = [curr_context, init_summ_idea, target_summ_idea,
                    persona.scratch.get_str_iss(), target_persona.scratch.get_str_iss(),
                    persona.name, target_persona.name, persona.name, target_persona.name,
                    persona.name]
    prompt = generate_prompt(prompt_input, "persona/prompt_template/v2/agent_chat_v1.txt")

    import json as _json
    def clean_up(raw, p=""):
        try:
            return _json.loads(raw)
        except:
            content = re.findall(r'\["([^"]+)",\s*"([^"]+)"\]', raw)
            return [[a, b] for a, b in content]

    def validate(val, p=""):
        return isinstance(val, list) and len(val) > 0

    output = run_prompt_str(prompt, clean_up, validate,
                            fail_safe=[[persona.name, "Hi!"], [target_persona.name, "Hi!"]],
                            repeat=5, verbose=verbose)
    return output, [output, prompt, None, prompt_input, None]


def run_gpt_generate_iterative_chat_utt_v2(maze, init_persona, target_persona,
                                            retrieved, curr_context, curr_chat,
                                            test_input=None, verbose=False):
    """Iterative single-utterance chat — uses original .txt template."""
    persona = init_persona
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat:
        for i in persona.a_mem.seq_chat:
            if i.object == target_persona.scratch.name:
                v1 = int((persona.scratch.curr_time - i.created).total_seconds() / 60)
                prev_convo_insert += f'{v1} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
                break
    if prev_convo_insert == "\n":
        prev_convo_insert = ""
    if persona.a_mem.seq_chat:
        if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds() / 60) > 480:
            prev_convo_insert = ""

    curr_sector = maze.access_tile(persona.scratch.curr_tile)['sector']
    curr_arena = maze.access_tile(persona.scratch.curr_tile)['arena']
    curr_location = f"{curr_arena} in {curr_sector}"

    retrieved_str = ""
    for key, vals in retrieved.items():
        for v in vals:
            retrieved_str += f"- {v.description}\n"

    convo_str = ""
    for i in curr_chat:
        convo_str += ": ".join(i) + "\n"
    if not convo_str:
        convo_str = "[The conversation has not started yet -- start it!]"

    init_iss = f"Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
    prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
                    curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
                    convo_str, init_persona.scratch.name, target_persona.scratch.name,
                    init_persona.scratch.name, init_persona.scratch.name, init_persona.scratch.name]
    prompt = generate_prompt(prompt_input, "persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt")

    import json as _json
    def _extract_first_json(s):
        start = s.find('{')
        end = s.find('}', start) + 1
        if start == -1 or end == 0:
            return None
        try:
            return _json.loads(s[start:end])
        except:
            return None

    def clean_up(raw, p=""):
        d = _extract_first_json(raw)
        if not d:
            return {"utterance": raw.strip(), "end": False}
        vals = list(d.values())
        result = {"utterance": vals[0] if vals else "...", "end": True}
        if len(vals) > 1 and ("f" in str(vals[1]).lower()):
            result["end"] = False
        return result

    def validate(val, p=""):
        return isinstance(val, dict) and "utterance" in val

    output = run_prompt_str(prompt, clean_up, validate,
                            fail_safe={"utterance": "...", "end": False},
                            repeat=3, verbose=verbose)
    return output, [output, prompt, None, prompt_input, None]
