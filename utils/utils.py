import subprocess
import os
import json
import logging
import concurrent.futures
import time
import re
import inspect
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
os.environ.get("ZHIPU_AI_API_KEY")
os.environ.get("LLAMA_API_KEY")

def init_client(cfg):
    global client
    if cfg.model.startswith("gpt"):
        from openai import OpenAI
        assert os.getenv('OPENAI_API_KEY') is not None, "Please set the environment variable OPENAI_API_KEY"
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    elif cfg.model.startswith("GLM"):
        from zhipuai import ZhipuAI 
        assert os.getenv('ZHIPU_AI_API_KEY') is not None, "Please set the environment variable ZHIPU_AI_API_KEY"
        zhipu_api_key = os.getenv('ZHIPU_AI_API_KEY')
        client = ZhipuAI(api_key=zhipu_api_key)
    else:
        # from openai import OpenAI
        # # We use llama api here. See the available models at https://docs.llama-api.com/quickstart#available-models
        # assert os.getenv('LLAMA_API_KEY') is not None, "Please set the environment variable LLAMA_API_KEY"
        # client = OpenAI(
        # api_key = os.getenv('LLAMA_API_KEY'),
        # base_url = "https://api.llama-api.com"
        # )
        assert os.getenv('LLAMA_API_KEY') is not None, "Please set the environment variable LLAMA_API_KEY"
        client = InferenceClient(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            token=os.getenv('LLAMA_API_KEY'),
        )

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if  len(log) > 0:
            if log_status and "Traceback" in log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string

def extract_messages(choices):
    messages = []
    current_message = ""
    for choice_list in choices:
        for choice in choice_list:
            current_message += choice.delta.content
    
    # Create the final message dictionary
    message_dict = {
        "finish_reason": choices[-1][0].finish_reason if choices and choices[-1] else None,
        "index": 0,
        "logprobs": None,
        "message": current_message
    }
    messages.append(message_dict)
    
    return messages

def multi_chat_completion(messages_list: list[list[dict]], n, model, temperature):
    """
    An example of messages_list:
    
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {"role": "system", "content": "You are a knowledgeable guide."},
            {"role": "user", "content": "How are you?"},
        ],
        [
            {"role": "system", "content": "You are a witty comedian."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    ]
    param: n: number of responses to generate for each message in messages_list
    """
    # If messages_list is not a list of list (i.e., only one conversation), convert it to a list of list
    assert isinstance(messages_list, list), "messages_list should be a list."
    if not isinstance(messages_list[0], list):
        messages_list = [messages_list]
    
    if len(messages_list) > 1:
        assert n == 1, "Currently, only n=1 is supported for multi-chat completion."
    
    if "gpt" not in model:
        # Transform messages if n > 1
        messages_list *= n
        n = 1

    with concurrent.futures.ThreadPoolExecutor() as executor:
        args = [(n, messages, model, temperature) for messages in messages_list]
        choices = executor.map(lambda p: chat_completion(*p), args)

    contents: list[str] = []
    for choice in choices:
        for c in choice:
            contents.append(c['message'])
    return contents


def chat_completion(n: int, messages: list[dict], model: str, temperature: float) -> list[dict]:
    """
    Generate n responses using OpenAI Chat Completions API
    """
    
    for attempt in range(1000):
        try:
            if "gpt" in model:
                response_cur = client.chat.completions.create(model=model, messages=messages, temperature=temperature, n=n)
            else:
                assert n == 1
                if "GLM" in model:
                    response_cur = client.chat.completions.create(model=model, messages=messages, temperature=min(temperature, 1.))
                elif "lama" in model:
                    response_cur = []
                    lama_messages = client.chat_completion(messages=messages, max_tokens=500, stream=True)
                    for lama_message in lama_messages:
                        response_cur.append(lama_message.choices)
                    response_cur = extract_messages(response_cur)
                else:
                    response_cur = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            break
        except Exception as e:
            logging.info(f"Attempt {attempt+1} failed with error: {e}")
            time.sleep(1)
    if response_cur is None:
        logging.info("Code terminated due to too many failed attempts!")
        exit()

    if "lama" in model:
        return response_cur
    else:
        return response_cur.choices

def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end+1])
    
    if code_string is None:
        return None
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name