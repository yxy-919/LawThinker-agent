from .base_agent import Agent
from .law_thinker_agent import LawThinker
from utils.register import register_class, registry
from collections import defaultdict
import re
import jsonlines
import json
from .prompts import get_response_prompt_report,get_law_thinker_instruction

@register_class(alias="Agent.Lawyer.GenerationBase")
class Lawyer_generation(LawThinker):
    def __init__(self, engine=None, lawyer_info=None, name="A"):
        super().__init__(engine=engine)
        self.name = name
        self.lawyer_greetings = "您好，我是您的法律顾问，请问有什么可以帮助您的？"
        self.engine = engine #记录驱动模型
        
        def default_value_factory():
            return [("system", self.system_prompt)]
        self.memories = defaultdict(default_value_factory) #当访问时不存在键的时候，会自动创建，以避免KeyError

    @staticmethod
    def add_parser_args(parser):
        pass
    
    def get_response(self, messages):
        response = self.engine.get_response(messages)
        return response

    def memorize(self, message, case_id):
        self.memories[case_id].append(message)
    
    def forget(self, case_id=None):
        def default_value_factory():
            return [("system", self.system_prompt)]
        if case_id is None:
            self.memories.pop(case_id)
        else:
            self.memories = defaultdict(default_value_factory)

    async def speak(self, content, case_id, semaphore, save_to_memory=True):
        memories = self.memories[case_id]

        messages = [{"role": memory[0], "content": memory[1]} for memory in memories]
        messages.append({"role": "user", "content": content})

        response = await self.engine.get_response(messages, semaphore)

        self.memorize(("user", content), case_id)
        self.memorize(("assistant", response), case_id)

        return response

    async def speak_with_thinker(self, content, case_id, turn, semaphore, save_to_memory=True):
        if not hasattr(self, 'memory_block'):
            self.memory_block = {"知识存储": [], "上下文存储": []}
        mems = self.memories[case_id]
        messages = [{"role": memory[0], "content": memory[1]} for memory in mems]
        messages.append({"role": "user", "content": content})
        history_text = "\n".join([f"{r}: {c}" for r, c in mems])
        prompt = get_law_thinker_instruction(task="Document Generation") + '\n'
        prompt += f"当前是第{turn}轮对话，历史对话：{history_text}\n用户的回复: {content}\n请向用户提问。"
        seq = {
            'finished': False,
            'prompt': prompt,
            'output': "",
            'messages': messages,
            'user_query': content,
            'system_prompt': self.system_prompt,
            'task': "Document Generation",
        }
        new_seq = await self.process_single_sequence(seq, semaphore)
        response_prompt = get_response_prompt_report(new_seq['system_prompt'], new_seq['output'], document_type="答辩状")
        messages = new_seq['messages']
        messages[0]['content'] = response_prompt
        generated_response = await self.engine.get_response(messages, semaphore)
        print(f"生成的对话内容: {generated_response}")
        new_seq['response'] = generated_response
        new_seq['output'] += generated_response
        response = new_seq['response']
        reasoning = new_seq['output']
        self.memorize(("user", content), case_id)
        self.memorize(("assistant", response), case_id)
        return response, reasoning

@register_class(alias="Agent.Lawyer.GPT_DD")
class GPTLawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.GPT4o_1120")(
            openai_api_key=args.lawyer_openai_api_key,
            openai_api_base=args.lawyer_openai_api_base,
            openai_model_name=args.lawyer_openai_model_name,
            temperature=args.lawyer_temperature,
            max_tokens=args.lawyer_max_tokens
        )
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(GPTLawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--lawyer_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--lawyer_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')

    def get_response(self, messages):
        response = self.engine.get_response(messages,  flag = 0)
        return response


@register_class(alias="Agent.Lawyer.Qwen3_14B_DD")
class Qwen3_14BLawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.qwen3_14B")()
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Qwen3_14BLawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--lawyer_top_p', type=float, default=1, help='top p')
        parser.add_argument('--lawyer_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--lawyer_presence_penalty', type=float, default=0, help='presence penalty')

    def get_response(self, messages):
        response = self.engine.get_response(messages)
        return response
    
@register_class(alias="Agent.Lawyer.Qwen3_8B_DD")
class Qwen3_8BLawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.qwen3_8B")()
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Qwen3_8BLawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--lawyer_top_p', type=float, default=1, help='top p')
        parser.add_argument('--lawyer_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--lawyer_presence_penalty', type=float, default=0, help='presence penalty')

    def get_response(self, messages):
        response = self.engine.get_response(messages)
        return response


@register_class(alias="Agent.Lawyer.Qwen3_32B_DD")
class Qwen3_32BLawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.qwen3_32B")()
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Qwen3_32BLawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--lawyer_top_p', type=float, default=1, help='top p')
        parser.add_argument('--lawyer_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--lawyer_presence_penalty', type=float, default=0, help='presence penalty')

    def get_response(self, messages):
        response = self.engine.get_response(messages)
        return response

@register_class(alias="Agent.Lawyer.Gemma12B_DD")
class Gemma12BLawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.Gemma12B")()
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Gemma12BLawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--lawyer_top_p', type=float, default=1, help='top p')
        parser.add_argument('--lawyer_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--lawyer_presence_penalty', type=float, default=0, help='presence penalty')

    def get_response(self, messages):
        response = self.engine.get_response(messages)
        return response


@register_class(alias="Agent.Lawyer.GLM9B_DD")
class GLM9BLawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.GLM9B")()
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(GLM9BLawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--lawyer_top_p', type=float, default=1, help='top p')
        parser.add_argument('--lawyer_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--lawyer_presence_penalty', type=float, default=0, help='presence penalty')

    def get_response(self, messages):
        response = self.engine.get_response(messages)
        return response


@register_class(alias="Agent.Lawyer.Chatlaw2_DD")
class Chatlaw2Lawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.Chatlaw2")()
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Chatlaw2Lawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--lawyer_top_p', type=float, default=1, help='top p')
        parser.add_argument('--lawyer_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--lawyer_presence_penalty', type=float, default=0, help='presence penalty')

    def get_response(self, messages):
        response = self.engine.get_response(messages)
        return response

    
@register_class(alias="Agent.Lawyer.LawLLM_DD")
class LawLLMLawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.lawllm")()
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(LawLLMLawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--lawyer_top_p', type=float, default=1, help='top p')
        parser.add_argument('--lawyer_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--lawyer_presence_penalty', type=float, default=0, help='presence penalty')

    def get_response(self, messages):
        response = self.engine.get_response(messages)
        return response

    
@register_class(alias="Agent.Lawyer.Deepseekv3_DD")
class Deepseekv3Lawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.deepseekv3")(
            openai_api_key=args.lawyer_openai_api_key,
            openai_api_base=args.lawyer_openai_api_base,
            openai_model_name=args.lawyer_openai_model_name,
            temperature=args.lawyer_temperature,
            max_tokens=args.lawyer_max_tokens
        )
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Deepseekv3Lawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--lawyer_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--lawyer_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')

    def get_response(self, messages):
        response = self.engine.get_response(messages,  flag = 0)
        return response
    
@register_class(alias="Agent.Lawyer.LLaMa_3_3_DD")
class LLaMa3_3Lawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.LLaMa3_3")()
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(LLaMa3_3Lawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--lawyer_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--lawyer_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')

    def get_response(self, messages):
        response = self.engine.get_response(messages,  flag = 0)
        return response
        

@register_class(alias="Agent.Lawyer.InternLM3_DD")
class InternLM3Lawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.InternLM3")()
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(InternLM3Lawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--lawyer_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--lawyer_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')

    def get_response(self, messages):
        response = self.engine.get_response(messages,  flag = 0)
        return response

    
@register_class(alias="Agent.Lawyer.Ministral8B_DD")
class Ministral8BLawyer_generation(Lawyer_generation):
    def __init__(self, args=None, lawyer_info=None, name="A"):
        engine = registry.get_class("Engine.Ministral8B")()
            
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.DD":
            system_prompt = ''
            profile = profiles["lawyer_DD"]
            template = '''
                                民事答辩状
                答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
                答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
                
                对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
                ......（写明答辩意见）
                证据和证据来源，证人姓名和住所：
                ......
            '''
            plaintiff_claim = lawyer_info['plaintiff_claim']
            plaintiff_case_details = lawyer_info['plaintiff_case_details']
            case_id = lawyer_info['court_info']['case_id']
            court_name = lawyer_info['court_info']['court_name']
            for p in profile:
                if '{court}' in p:
                    system_prompt += p.format(court = court_name) + '\n'
                elif '{template}' in p:
                    system_prompt += p.format(template = template) + '\n'
                elif '{case_details}' in p:
                    system_prompt += p.format(case_details = plaintiff_case_details) + '\n'
                elif '{claims}' in p:
                    system_prompt += p.format(claims = '；'.join(plaintiff_claim)) + '\n'
                elif '{case_id}' in p:
                    system_prompt += p.format(case_id = case_id) + '\n'
                else:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Ministral8BLawyer_generation, self).__init__(engine, lawyer_info, name)
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--lawyer_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--lawyer_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--lawyer_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--lawyer_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--lawyer_max_tokens', type=int, default=4096, help='max tokens')

    def get_response(self, messages):
        response = self.engine.get_response(messages)
        return response

    