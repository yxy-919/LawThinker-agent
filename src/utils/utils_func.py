import json
from openai import OpenAI,AsyncOpenAI
from requests.exceptions import ConnectionError, Timeout, RequestException
import os
import asyncio

# Set vllm serve api and url
# 部署环境的vllm serve api和url
vllm_api_url1 = {
    'base_url': "http://localhost:19001/v1",
    "api_key": "EMPTY"
}

# 部署重要角色的vllm serve api和url
vllm_api_url2 = {
    'base_url': "http://localhost:19003/v1",
    "api_key": "EMPTY"
}

# gpt-4o
api_key = ''
api_base = ''


os.environ['OPENAI_API_KEY'] = api_key
os.environ['OPENAI_API_BASE'] = api_base

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_json(data, save_path):
    # 修改
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 修改
# 用qwen3-32b提取KQ、LC的topic 工具
async def get_completion_extract(semaphore: asyncio.Semaphore, prompt, history, flag):
    SYSTEM_PROMPT = """你是一个得力的助手。"""
    messages = []
    messages.append({'role': 'system', "content": SYSTEM_PROMPT})
    if history != []:
        for h in history:
            messages.append({'role':'user',"content":h[0]})
            messages.append({'role':'assistant',"content":h[1]})
    else:
        messages.append({'role':'user',"content":prompt})

    client = AsyncOpenAI(
            api_key = "EMPTY",
            base_url = "http://localhost:19001/v1"
        )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with semaphore:
                chat_completion = await client.chat.completions.create(
                    messages=messages,
                    model="Qwen2.5-32B",
                    max_tokens=4096,
                    temperature=0.0
                    )
                response = chat_completion.choices[0].message.content
                history.append((prompt, response))
                return response, history
        
        except (ConnectionError, Timeout) as e:
            print(f"Network error occurred: {e}. Retrying {attempt + 1}/{max_retries}...")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1 * (attempt + 1))
            
        except RequestException as e:
            print(f"An error occurred: {e}.")
            raise 
        
        except Exception as e:
            print(e)
            
    return "Unable to get a response after several attempts."

 
#  用gpt-4o进行评测
def get_completion(prompt, history, flag):
    client = OpenAI(api_key=api_key,
                    base_url=api_base)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            SYSTEM_PROMPT = """你是一个得力的助手。"""
            messages = []
            messages.append({'role': 'system', "content": SYSTEM_PROMPT})
            if history != []:
                for h in history:
                    messages.append({'role':'user',"content":h[0]})
                    messages.append({'role':'assistant',"content":h[1]})
            else:
                messages.append({'role':'user',"content":prompt})
        
            if flag == 1:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="gpt-4o-2024-08-06",
                    response_format={"type": "json_object"},
                    max_tokens=4096,
                    temperature=0
                    )
            else:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="gpt-4o-2024-08-06",
                    max_tokens=4096,
                    temperature=0
                    )
            
            response = chat_completion.choices[0].message.content
            history.append((prompt, response))
            return response, history
        
        except (ConnectionError, Timeout) as e:
            print(f"Network error occurred: {e}. Retrying {attempt + 1}/{max_retries}...")
            if attempt == max_retries - 1:
                raise
                
        except RequestException as e:
            print(f"An error occurred: {e}.")
            raise 
        except Exception as e:
            print(e)
            
    return "Unable to get a response after several attempts."