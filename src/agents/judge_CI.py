from .base_agent import Agent
from agents.law_thinker_agent import LawThinker
from utils.register import registry, register_class
import json
import threading
from .prompts import get_response_prompt_judge,get_law_thinker_instruction
import asyncio

@register_class(alias="Agent.Judge.civilPredictionBase")
class Judge_civilPrediction(LawThinker):
    def __init__(self, engine=None, judge_info=None, name="B"):
        super().__init__(engine=engine)
        self.name = name
        self.engine = engine
        self.memories = [("system", self.system_prompt)]
        
    @staticmethod
    def add_parser_args(parser):
        pass
    
    def get_response(self, messages):
        response = self.engine.get_response(messages)
        return response
    
    def memorize(self, message):
        self.memories.append(message)
    
    async def speak(self, content, semaphore, save_to_memory = True):
        messages = [{"role": memory[0], "content": memory[1]} for memory in self.memories]
        messages.append({"role": "user", "content": f"<Judge> {content}"})

        response = await self.engine.get_response(messages, semaphore)
        
        if save_to_memory:
            self.memorize(("user", f"<Judge> {content}"))
            self.memorize(("assistant", response))
        
        return response

    async def speak_with_thinker(self, content, turn, semaphore, save_to_memory=True):
        messages = [{"role": memory[0], "content": memory[1]} for memory in self.memories]
        messages.append({"role": "user", "content": f"{content}"})
        history_text = "\n".join([f"{r}: {c}" for r, c in self.memories])
        prompt = get_law_thinker_instruction(task="Court Simulation") + '\n'
        prompt += f"当前是第{turn}轮对话，历史对话：{history_text}\n回复: {content}\n请提问。"
        seq = {
            'finished': False,
            'prompt': prompt,
            'output': "",
            'messages': messages,
            'user_query': content,
            'system_prompt': self.system_prompt,
            'task': "Court Simulation",
        }
        new_seq = await self.process_single_sequence(seq, semaphore)
        response_prompt = get_response_prompt_judge(new_seq['system_prompt'], new_seq['output'], court_type="民事法庭")
        messages = new_seq['messages']
        messages[0]['content'] = response_prompt
        generated_response = await self.engine.get_response(messages, semaphore)
        print(f"生成的对话内容: {generated_response}")
        new_seq['response'] = generated_response
        new_seq['output'] += generated_response
        response = new_seq['response']
        reasoning = new_seq['output']
        if save_to_memory:
            self.memorize(("user", f"{content}"))
            self.memorize(("assistant", response))
        return response, reasoning
    
@register_class(alias="Agent.Judge.GPT_CI")
class GPTJudge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.GPT4o_1120")(
            openai_api_key=args.judge_openai_api_key,
            openai_api_base=args.judge_openai_api_base,
            openai_model_name=args.judge_openai_model_name,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens
        )
        
        # 编写profile
        id = judge_info['id']
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(GPTJudge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--judge_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--judge_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')



@register_class(alias="Agent.Judge.Qwen3_14B_CI")
class Qwen3_14BJudge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.qwen3_14B")()
        
        # 编写profile
        id = judge_info['id']
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        # 修改
                        elif '你在庭审过程中应遵守以下注意事项' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        # 修改
                        elif '你在庭审过程中应遵守以下注意事项' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        # 修改
                        elif '你在庭审过程中应遵守以下注意事项' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        # 修改
                        elif '你在庭审过程中应遵守以下注意事项' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(Qwen3_14BJudge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')
    
    
    
@register_class(alias="Agent.Judge.Qwen3_32B_CI")
class Qwen3_32BJudge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.qwen3_32B")()
        
        # 编写profile
        id = judge_info['id']
        print(id)
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        # plaintiff_info = judge_info["plaintiff_info"]
        # defendant_info = judge_info["defendant_info"]
        # third_party_findings = judge_info["third_party_findings"]
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        # 修改
                        elif '你在庭审过程中应遵守以下注意事项' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        # 修改
                        elif '你在庭审过程中应遵守以下注意事项' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        # 修改
                        elif '你在庭审过程中应遵守以下注意事项' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        # 修改
                        elif '你在庭审过程中应遵守以下注意事项' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(Qwen3_32BJudge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')



@register_class(alias="Agent.Judge.Gemma12B_CI")
class Gemma12BJudge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.Gemma12B")()
        
        # 编写profile
        id = judge_info['id']
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(Gemma12BJudge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')
    

@register_class(alias="Agent.Judge.GLM9B_CI")
class GLM9BJudge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.GLM9B")()
        
        # 编写profile
        id = judge_info['id']
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(GLM9BJudge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')





@register_class(alias="Agent.Judge.Chatlaw2_CI")
class Chatlaw2Judge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.Chatlaw2")()
        
        # 编写profile
        id = judge_info['id']
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(Chatlaw2Judge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')
    

    
    
    
@register_class(alias="Agent.Judge.Deepseekv3_CI")
class Deepseekv3Judge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.deepseekv3")(
            openai_api_key=args.judge_openai_api_key,
            openai_api_base=args.judge_openai_api_base,
            openai_model_name=args.judge_openai_model_name,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens
        )
        
        # 编写profile
        id = judge_info['id']
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(Deepseekv3Judge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--judge_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--judge_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')


@register_class(alias="Agent.Judge.LlaMa3_3_CI")
class LLaMa3_3Judge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.LLaMa3_3")()
        
        # 编写profile
        id = judge_info['id']
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(LLaMa3_3Judge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--judge_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--judge_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
    

@register_class(alias="Agent.Judge.InternLM3_CI")
class InternLM3Judge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.InternLM3")()
        
        # 编写profile
        id = judge_info['id']
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(InternLM3Judge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--judge_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--judge_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')


    
@register_class(alias="Agent.Judge.Ministral8B_CI")
class Ministral8BJudge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.Ministral8B")()
        
        # 编写profile
        id = judge_info['id']
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(Ministral8BJudge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--judge_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--judge_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
    
    
    
@register_class(alias="Agent.Judge.LawLLM_CI")
class LawLLMJudge_civilPrediction(Judge_civilPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.lawllm")()
        
        # 编写profile
        id = judge_info['id']
        plaintiff_info = judge_info['specific_characters']['plaintiff']
        defendant_info = judge_info['specific_characters']['defendant']
        third_party_findings = judge_info['court_information']['third_party_findings']
        if len(third_party_findings) == 0:
            third_party_findings = '无'
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CI":
            profile = profiles['judge_CI']
            system_prompt = ''
            if "gender" in plaintiff_info.keys():
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_name}' in p:
                            system_prompt += p.format(plaintiff_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_sex}' in p:
                            system_prompt += p.format(plaintiff_sex = plaintiff_info['gender']) + '\n'
                        elif '{plaintiff_birth}' in p:
                            system_prompt += p.format(plaintiff_birth = plaintiff_info['birth_date']) + '\n'
                        elif '{plaintiff_nation}' in p:
                            system_prompt += p.format(plaintiff_nation = plaintiff_info['ethnicity']) + '\n'
                        elif '{plaintiff_address}' in p:
                            system_prompt += p.format(plaintiff_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            else:
                if "gender" in defendant_info.keys():
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_name}' in p:
                            system_prompt += p.format(defendant_name = defendant_info['name']) + '\n'
                        elif '{defendant_sex}' in p:
                            system_prompt += p.format(defendant_sex = defendant_info['gender']) + '\n'
                        elif '{defendant_birth}' in p:
                            system_prompt += p.format(defendant_birth = defendant_info['birth_date']) + '\n'
                        elif '{defendant_nation}' in p:
                            system_prompt += p.format(defendant_nation = defendant_info['ethnicity']) + '\n'
                        elif '{defendant_address}' in p:
                            system_prompt += p.format(defendant_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
                else:
                    for p in profile:
                        if '{plaintiff_company_name}' in p:
                            system_prompt += p.format(plaintiff_company_name = plaintiff_info['name']) + '\n'
                        elif '{plaintiff_company_legal_person}' in p:
                            system_prompt += p.format(plaintiff_company_legal_person = plaintiff_info['representative']) + '\n'
                        elif '{plaintiff_company_address}' in p:
                            system_prompt += p.format(plaintiff_company_address = plaintiff_info['address']) + '\n'
                        elif '{defendant_company_name}' in p:
                            system_prompt += p.format(defendant_company_name = defendant_info['name']) + '\n'
                        elif '{defendant_company_legal_person}' in p:
                            system_prompt += p.format(defendant_company_legal_person = defendant_info['representative']) + '\n'
                        elif '{defendant_company_address}' in p:
                            system_prompt += p.format(defendant_company_address = defendant_info['address']) + '\n'
                        elif '{facts}' in p:
                            system_prompt += p.format(facts = third_party_findings) + '\n'
                        elif '同时，你需要遵循如下的注意事' in p:
                            system_prompt += '\n' + p + '\n'
                        elif '{' not in p:
                            system_prompt += p + '\n'
            
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            system_prompt = system_prompt.replace('\n\n\n', '\n\n')
            self.system_prompt = system_prompt
            
        super(LawLLMJudge_civilPrediction, self).__init__(engine)
        self.id = id
        self.judge_greetings = "现在开庭。"
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--judge_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--judge_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
