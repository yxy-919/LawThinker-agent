


from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
import os
from abc import abstractmethod
import sys
from typing import List
import asyncio
from openai import OpenAI, AsyncOpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from utils.register import register_class
from utils.utils_func import vllm_api_url1
from .base_engine import Engine
import time
import torch._dynamo
import json
from requests.exceptions import ConnectionError, Timeout, RequestException
import os

torch._dynamo.config.suppress_errors = True

@register_class(alias="Engine.qwen3_32B")
class qwen3_32BEngine(Engine):
    _instance = None 

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("Creating new qwen3_32BEngine instance", flush=True)
            cls._instance = super(qwen3_32BEngine, cls).__new__(cls)
            cls._instance._initialized = False
        else:
            print("Reusing existing qwen3_32BEngine instance", flush=True)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.model_path = "model/Qwen3-32B"
        self.api_key = vllm_api_url1['api_key']
        self.base_url = vllm_api_url1['base_url']
        
        self.client = AsyncOpenAI(
            api_key = self.api_key,
            base_url = self.base_url
        )
    
    # ============ Chat 模式 ============
    async def get_response(self, messages, semaphore: asyncio.Semaphore, stop: List[str]=None):
        max_retries = 3
        for attempt in range(max_retries): 
            try:
                async with semaphore:
                    chat_completion = await self.client.chat.completions.create(
                        messages=messages,
                        model="Qwen3-32B",
                        temperature=0.7,
                        top_p=0.8,
                        max_tokens=32768,
                        stop=stop,
                        extra_body={
                            "repetition_penalty": 1.0,
                            "top_k": 20,
                            "include_stop_str_in_output": True,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                        timeout=1500,
                    )
                    return chat_completion.choices[0].message.content
            except Exception as e:
                print(f"Generate Response Error occurred: {e}, Starting retry attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return ""
                await asyncio.sleep(1 * (attempt + 1))
        return ""
    
    # ============ Completion / Tool 模式 ============
    async def get_response_with_tool(self, prompt: str, semaphore: asyncio.Semaphore, stop: List[str]=None) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    response = await self.client.completions.create(
                        model="Qwen3-32B",
                        prompt=prompt,
                        temperature=0.7,
                        top_p=0.8,
                        max_tokens=32768,
                        stop=stop,
                        extra_body={
                            "repetition_penalty": 1.0,
                            "top_k": 20,
                            "include_stop_str_in_output": True,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                        timeout=1500,
                    )
                    return response.choices[0].text
            except Exception as e:
                print(f"Generate completion error: {e}. Retrying {attempt + 1}/{max_retries}...")
                if attempt == max_retries - 1:
                    return ""
                await asyncio.sleep(1 * (attempt + 1))
        return ""