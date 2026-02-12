# 注册不同的Engine
from .gpt_4o_1120 import GPT_1120Engine
from .lawllm import LawLLMEngine
from .deepseek_v3 import DeepseekEngine
from .ministral8B import Ministral8BEngine
from .GLM_4_9B import GLM9BEngine
from .chatlaw2 import Chatlaw2Engine
from .qwen3_14B import qwen3_14BEngine
from .qwen3_32B import qwen3_32BEngine
from .gemma12b import Gemma12BEngine
from .internlm3 import InternLM3Engine
from .llama33_70B import LLaMa3_3Engine
from .qwen3_4B import qwen3_4BEngine

__all__= [
    'GPT_1120Engine',
    'LawLLMEngine',
    'DeepseekEngine',
    'Ministral8BEngine',
    'GLM9BEngine',
    'Chatlaw2Engine',
    'qwen3_14BEngine',
    'qwen3_32BEngine',
    'Gemma12BEngine',
    'InternLM3Engine',
    'LLaMa3_3Engine',
    'qwen3_4BEngine'
]