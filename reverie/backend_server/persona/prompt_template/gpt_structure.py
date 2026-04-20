"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling LLM APIs.

Refactored to use LangChain + LiteLLM gateway. All function signatures are
preserved so that callers (run_gpt_prompt.py, cognitive modules) need zero
changes.
"""
import json
import random
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
  from utils import *
except ModuleNotFoundError:
  debug = True

from llm_config import (LITELLM_BASE_URL, LITELLM_API_KEY,
                         LLM_MODEL, EMBEDDING_MODEL,
                         LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY,
                         LANGFUSE_HOST)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Langfuse observability — auto-traces every LLM call
# ---------------------------------------------------------------------------
_langfuse_handler = None
try:
  from langfuse.langchain import CallbackHandler
  _langfuse_handler = CallbackHandler(
      secret_key=LANGFUSE_SECRET_KEY,
      public_key=LANGFUSE_PUBLIC_KEY,
      host=LANGFUSE_HOST,
  )
except ImportError:
  pass

# ---------------------------------------------------------------------------
# Shared LLM and Embedding instances, pointed at the LiteLLM gateway.
# ---------------------------------------------------------------------------
_llm_callbacks = [_langfuse_handler] if _langfuse_handler else []

_llm = ChatOpenAI(
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model=LLM_MODEL,
    callbacks=_llm_callbacks,
)

_embeddings = OpenAIEmbeddings(
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model=EMBEDDING_MODEL,
)


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def ChatGPT_single_request(prompt):
  temp_sleep()
  try:
    return _llm.invoke(prompt).content
  except Exception as e:
    print(f"ChatGPT ERROR: {e}")
    return "ChatGPT ERROR"


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
  """
  Given a prompt, make a request via the LiteLLM gateway and return the
  response string.  Model is configured in llm_config.py.
  """
  temp_sleep()
  try:
    return _llm.invoke(prompt).content
  except Exception as e:
    print(f"ChatGPT ERROR: {e}")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt):
  """
  Given a prompt, make a request via the LiteLLM gateway and return the
  response string.  Model is configured in llm_config.py.
  """
  try:
    return _llm.invoke(prompt).content
  except Exception as e:
    print(f"ChatGPT ERROR: {e}")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):

    try:
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass

  return False


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):

    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose:
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass
  print ("FAIL SAFE TRIGGERED")
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter):
  """
  Legacy Completion-style call.  Converted to Chat API via LangChain since
  the old text-davinci models are deprecated.

  Note: max_tokens from the legacy parameter dict is intentionally NOT passed
  through, because modern reasoning models (e.g. gpt-5-mini) allocate
  reasoning tokens from the same budget, leaving no room for output.
  """
  temp_sleep()
  try:
    bind_kwargs = {}
    temperature = gpt_parameter.get("temperature")
    if temperature is not None:
      bind_kwargs["temperature"] = temperature
    stop = gpt_parameter.get("stop")
    if stop:
      bind_kwargs["stop"] = stop

    if bind_kwargs:
      llm_with_params = _llm.bind(**bind_kwargs)
    else:
      llm_with_params = _llm
    return llm_with_params.invoke(prompt).content
  except Exception as e:
    print(f"TOKEN LIMIT EXCEEDED: {e}")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file):
  """
  Takes in the current input (e.g. comment that you want to classifiy) and
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this
  function replaces this substr with the actual curr_input to produce the
  final promopt that will be sent to the GPT3 server.
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file.
  RETURNS:
    a str prompt that will be sent to OpenAI's GPT server.
  """
  if type(curr_input) == type("string"):
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
  if verbose:
    print (prompt)

  for i in range(repeat):
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose:
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model=None):
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"
  return _embeddings.embed_query(text)


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50,
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0,
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response):
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1:
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt,
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)




















