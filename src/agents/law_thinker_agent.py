from .base_agent import Agent
from utils.register import register_class
from collections import defaultdict
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import threading
import asyncio
# 工具统一导入
from tools import law_tool, crime_tool, check_tool, procedure_tool, template_tool, memory_tool, plan_tool, case_tool
from .prompts import get_deep_analysis_prompt
from sentence_transformers import SentenceTransformer 

@register_class(alias="Agent.LawThinker")
class LawThinker(Agent):
    # 全局共享的 BGE 实例（进程内单例）
    _shared_bge_model = None
    _shared_bge_path = None
    _bge_lock = threading.Lock()
    _bge_log_once = False

    def __init__(self, engine=None, bge_model_path=None):
        self.engine = engine
        def default_value_factory():
            return [("system", self.system_prompt)]
        self.memories = defaultdict(default_value_factory) 
        self.memory_block: Dict[str, List[Any]] = {"知识存储": [], "上下文存储": []}
        self.bge_model_path = bge_model_path if bge_model_path is not None else "../model/BAAI_bge-m3"
        self._bge_model = None
        
    # ---------- 基础封装 ----------
    def _ensure_bge_model(self):
        """
        进程内单例加载 BGE：
        - 所有 LawThinker 实例共享同一个 SentenceTransformer 实例
        - 避免重复加载/占用显存
        """
        if SentenceTransformer is None:
            return None

        # 已有实例先复用
        if self._bge_model is not None:
            return self._bge_model
        if LawThinker._shared_bge_model is not None:
            self._bge_model = LawThinker._shared_bge_model
            return self._bge_model

        # 双重检查 + 互斥锁，保证只加载一次
        with LawThinker._bge_lock:
            if LawThinker._shared_bge_model is not None:
                self._bge_model = LawThinker._shared_bge_model
                return self._bge_model
            try:
                model_path = getattr(self, 'bge_model_path', None) or "../model/BAAI_bge-m3"
                LawThinker._shared_bge_model = SentenceTransformer(model_path)
                LawThinker._shared_bge_path = model_path
                self._bge_model = LawThinker._shared_bge_model
                if not LawThinker._bge_log_once:
                    print(f"BGE模型加载成功（共享），路径：{model_path}")
                    LawThinker._bge_log_once = True
            except Exception as e:
                print(f"BGE模型加载失败，路径：{getattr(self, 'bge_model_path', None)}，错误信息：{e}")
                LawThinker._shared_bge_model = None
                self._bge_model = None
        return self._bge_model

    def memorize(self, message):
        self.memories.append(message)

    def forget(self):
        self.memories = [("system", self.system_prompt)]
        # 长期记忆 self.memory_block 保留

    # ---------- 工具执行层 ----------
    async def _execute_tool(self, name: str, arguments: Dict[str, Any], tool_calls_semaphore: asyncio.Semaphore) -> Any:
        # 先做参数完整性校验，缺参时直接返回提示，避免 KeyError
        args = arguments or {}
        required_args = {
            "law_retrieval": ["query", "topk"],
            "law_recommendation": ["law"],
            "charge_expansion": ["charges"],
            "procedure_retrieval": ["court_type", "stage"],
            "case_retrieval": ["type", "query"],
            "law_check": ["law_name"],
            "fact_law_relevance_check": ["fact", "law"],
            "crime_law_consistency_check": ["crime", "law"],
            "memory_store": ["memory_type", "content"],
            "memory_fetch": ["memory_type"],
            "plan_generation": ["document_type"],
            "template_retrieval": ["template_type"],
            "document_format_check": ["document_type", "document"],
            "law_query_rewrite": ["query", "context"],
            "procedure_check": ["court_type"],
        }
        missing = [
            key for key in required_args.get(name, [])
            if args.get(key) in (None, "")
        ]
        if missing:
            return f"工具 {name} 缺少必要参数：{', '.join(missing)}。请按规范重新调用。"

        bge = self._ensure_bge_model()
        if name == "law_retrieval":
            return law_tool.law_retrieval(args["query"], int(args["topk"]), bge)
        if name == "law_recommendation":
            return law_tool.law_recommendation(args["law"], bge)
        if name == "charge_expansion":
            return crime_tool.charge_expansion(args["charges"], bge)
        if name == "procedure_retrieval":
            return procedure_tool.procedure_retrieval(args["court_type"], int(args["stage"]))
        if name == "case_retrieval":
            return case_tool.case_retrieval(args["type"], args["query"])
        if name == "law_check":
            return law_tool.law_check(args["law_name"], bge)
        if name == "fact_law_relevance_check":
            return await check_tool.fact_law_relevance_check(args["fact"], args["law"], bge, tool_calls_semaphore)
        if name == "crime_law_consistency_check":
            return check_tool.crime_law_consistency_check(args["crime"], args["law"], bge)
        if name == "memory_store":
            return memory_tool.storeMemory(args["memory_type"], args["content"], self.memory_block)
        if name == "memory_fetch":
            return memory_tool.fetchMemory(args["memory_type"], self.memory_block)
        if name == "plan_generation":
            return plan_tool.plan_generation(args["document_type"])
        if name == "template_retrieval":
            return template_tool.template_retrieval(args["template_type"])
        if name == "document_format_check":
            return template_tool.document_format_check(args["document_type"], args["document"])
        if name == "law_query_rewrite":
            return await check_tool.law_query_rewrite(args["query"], args["context"], tool_calls_semaphore)
        if name == "procedure_check":
            return procedure_tool.procedure_check(args["court_type"])
        return f"未知工具：{name}，请按照正确的格式调用工具。\n"

    TOOL_ZH = {
        "law_check": "法条内容核查",
        "fact_law_relevance_check": "事实-法条相关性检查",
        "crime_law_consistency_check": "罪名-法条一致性检查",
        "memory_store": "记忆存储",
        "memory_fetch": "记忆读取",
        "document_format_check": "文书格式检查",
        "law_query_rewrite": "法条查询语句改写",
        "procedure_check": "流程检查",
    }

    def summarize_tool_results(self, output: str) -> str:
        """
        从 deep_analysis 的 output 中提取工具调用及结果，
        重新组织成中文条目式总结。
        """
        # ① 提取所有 <tool_call>…</tool_call> + <tool_call_result>…</tool_call_result>
        pattern = re.compile(
            r"<tool_call>(.*?)</tool_call>.*?<tool_call_result>(.*?)</tool_call_result>",
            re.S
        )
        matches = pattern.findall(output)
        if not matches:
            return ""

        summary_lines = ["以下是我的核查结果："]
        for idx, (call_json, result_text) in enumerate(matches, 1):
            # 解析 JSON
            try:
                call = json.loads(call_json.strip())
                name = call.get("name", "").strip()
                args = call.get("arguments", {})
            except json.JSONDecodeError:
                name, args = "未知工具", {}
            # 中译工具名
            zh_name = self.TOOL_ZH.get(name, name)
            # 参数简化显示
            args_str = json.dumps(args, ensure_ascii=False) if args else "无"
            result_text = result_text.strip()
            summary_lines.append(
                f"（{idx}）我使用了「{zh_name}」工具（参数：{args_str}），得到的结果是：{result_text}"
            )

        return "\n".join(summary_lines)
        
    # ---------- 推理循环 ----------
    def extract_between(self, text, start_marker, end_marker):
        """Extracts text between two markers in a string."""
        try:
            pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
            # Run pattern matching with timeout
            matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
            if matches:
                return matches[0][::-1].strip()
            return None
        except Exception as e:
            print(f"---Error:---\n{str(e)}")
            print(f"-------------------")
            return None

    async def deep_analysis(self, reasoning_process: str, user_query: str, tool_used: str, task: str, semaphore, tool_calls_semaphore) -> str:
        """
        - reasoning_process: 当前探索阶段的推理过程
        - user_query: 用户的问题或回复
        - tool_used: 使用的工具
        对每一次探索得到的内容进行check、总结、保存
        """
        print("开始deep analysis:")
        prompt = get_deep_analysis_prompt(reasoning_process,user_query,tool_used, task)
        output = ""
        total_tokens = len(prompt.split())
        MAX_TOKENS = 30000
        MAX_INTERACTIONS = 3
        total_interactions = 0
        while True:
            print("进入while循环内")
            response = await self.engine.get_response_with_tool(
                prompt=prompt,
                semaphore=semaphore,
                stop=["</tool_call>"]
            )
            print(f"response: {response}")
            output += response
            prompt += response
            total_tokens += len(response.split())
            if total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS or not response.rstrip().endswith("</tool_call>"):
                break
            
            if response.rstrip().endswith("</tool_call>"):
                total_interactions += 1
                tool_call_str = self.extract_between(response, "<tool_call>", "</tool_call>")
                if tool_call_str:
                    try:
                        tool_call = json.loads(tool_call_str)
                        print(f"tool_call: {tool_call}")
                        name = tool_call.get("name")
                        args = tool_call.get("arguments", {}) or {}
                        result = await self._execute_tool(name, args, tool_calls_semaphore) if name else ""
                    except Exception as e:
                        result = "请按照正确的格式调用工具。"
                    print(f"result: {result}")
                    # Append tool call results
                    tool_call_result = f"\n<tool_call_result>\n{result}\n</tool_call_result>\n"
                    output += tool_call_result
                    prompt += tool_call_result
                    total_tokens += len(tool_call_result.split())
                else:
                    tool_call_result = f"\n<tool_call_result>\n请按照正确的格式调用工具。\n</tool_call_result>\n"
                    output += tool_call_result
                    prompt += tool_call_result
                    total_tokens += len(tool_call_result.split())
            else:
                break
        return output
        
    async def process_single_sequence(
        self,
        seq: Dict,
        semaphore,
        tool_calls_semaphore,
        turn: int = 0,
    ) -> Dict:
        """Process a single sequence through its entire reasoning chain"""
        # 定义工具调用的信号量
        # max_tool_calls = 5
        # tool_calls_semaphore = asyncio.Semaphore(max_tool_calls)
        total_tokens = len(seq['prompt'].split())

        seq['deep_analysis'] = []
        # First response uses completion mode
        response = await self.engine.get_response_with_tool(prompt=seq['prompt'], semaphore=semaphore, stop=["</tool_call>"])
        tokens_this_response = len(response.split())
        total_tokens += tokens_this_response

        seq['output'] += response
        seq['prompt'] += response
        last_response = response

        while not seq['finished']:
            if not seq['output'].rstrip().endswith("</tool_call>") or turn >= 5:
                seq['finished'] = True
                break

            if last_response.rstrip().endswith("</tool_call>"):
                turn += 1
                tool_call_str = self.extract_between(last_response, "<tool_call>", "</tool_call>")
                print(f"tool_call: {tool_call_str}")
                if tool_call_str:
                    try:
                        call = json.loads(tool_call_str)
                        name = call.get("name")
                        args = call.get("arguments", {}) or {}
                        # 增加信号量
                        result = await self._execute_tool(name, args, tool_calls_semaphore)
                        print(f"result: {result}")
                        # deep analysis
                        used_tool = f"使用的工具名称：{name}，参数：{json.dumps(args, ensure_ascii=False)}；工具调用的结果：{result}"
                        reasoning_process = last_response
                        # 记忆为空时不需要check
                        if name == "memory_fetch" and (result == "知识存储：" or result == "上下文存储："):
                            analysis = "当前记忆为空。"
                            output = ""
                        else:
                            output = await self.deep_analysis(reasoning_process, seq['user_query'], used_tool, seq['task'], semaphore, tool_calls_semaphore)
                            analysis = self.summarize_tool_results(output)
                            print(f"总结核查结果:  {analysis}")
                            # generate_analysis_prompt = f"你是一名善于总结的助手，请根据以下内容总结核查的结果：\n{output}\n。如果其中调用了法条查询语句改写工具，请在核查结果中说明改写后的查询语句，并说明改写的原因。请严格按照以下格式输出总结内容：\n<check_result>总结核查的结果</check_result>"
                            # generate_analysis = await self.engine.get_response_with_tool(prompt=generate_analysis_prompt, semaphore=semaphore, stop=["</check_result>"])
                            # print(f"总结核查结果: {generate_analysis}")
                            # analysis = self.extract_between(generate_analysis, "<check_result>", "</check_result>")
                            # if analysis == None:
                            #     analysis = generate_analysis


                        seq['deep_analysis'].append(
                            {
                                "reasoning_process": reasoning_process,
                                "user_query": seq['user_query'],
                                "used_tool": used_tool,
                                "output": output,
                                "analysis": analysis
                            }
                        )
                        # update prompt and output
                        append_text = f"\n<tool_call_result>\n{result}\n{analysis}\n</tool_call_result>\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        total_tokens += len(append_text.split())
                        
                    except Exception as e:
                        print(f"error: {e}")
                        append_text = f"\n<tool_call_result>\n请按照正确的格式调用工具。\n</tool_call_result>\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        total_tokens += len(append_text.split())
                else:
                    append_text = f"\n<tool_call_result>\n请按照正确的格式调用工具。\n</tool_call_result>\n"
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    total_tokens += len(append_text.split())
                response = await self.engine.get_response_with_tool(prompt=seq['prompt'], semaphore=semaphore, stop=["</tool_call>"])
                tokens_this_response = len(response.split())
                total_tokens += tokens_this_response

                seq['output'] += response
                seq['prompt'] += response
                last_response = response
            else:
                break
        return seq
