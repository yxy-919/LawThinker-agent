from .base_agent import Agent
from agents.law_thinker_agent import LawThinker
from .prompts import get_law_thinker_instruction, get_response_prompt_judge
from utils.register import registry, register_class
import json

@register_class(alias="Agent.Judge.criminalPredictionBase")
class Judge_criminalPrediction(LawThinker):
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
        response_prompt = get_response_prompt_judge(new_seq['system_prompt'], new_seq['output'], court_type="刑事法庭")
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


@register_class(alias="Agent.Judge.GPT_CR")
class GPTJudge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.GPT4o_1120")(
            openai_api_key=args.judge_openai_api_key,
            openai_api_base=args.judge_openai_api_base,
            openai_model_name=args.judge_openai_model_name,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens
        )
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(GPTJudge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--judge_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--judge_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
    

@register_class(alias="Agent.Judge.Qwen3_14B_CR")
class Qwen3_14BJudge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.qwen3_14B")()
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Qwen3_14BJudge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')
    

@register_class(alias="Agent.Judge.Qwen3_32B_CR")
class Qwen3_32BJudge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.qwen3_32B")()
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Qwen3_32BJudge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')
    
    
@register_class(alias="Agent.Judge.Gemma12B_CR")
class Gemma12BJudge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.Gemma12B")()
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Gemma12BJudge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')



@register_class(alias="Agent.Judge.GLM9B_CR")
class GLM9BJudge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.GLM9B")()
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(GLM9BJudge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')
    


@register_class(alias="Agent.Judge.Chatlaw2_CR")
class Chatlaw2Judge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.Chatlaw2")()
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Chatlaw2Judge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')
    
    
    
@register_class(alias="Agent.Judge.Deepseekv3_CR")
class Deepseekv3Judge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.deepseekv3")(
            openai_api_key=args.judge_openai_api_key,
            openai_api_base=args.judge_openai_api_base,
            openai_model_name=args.judge_openai_model_name,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens
        )
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Deepseekv3Judge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--judge_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--judge_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
    

@register_class(alias="Agent.Judge.LawLLM_CR")
class LawLLMJudge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.lawllm")()
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(LawLLMJudge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')
    

@register_class(alias="Agent.Judge.InternLM3_CR")
class InternLM3Judge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.InternLM3")()
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(InternLM3Judge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')

@register_class(alias="Agent.Judge.LLaMa3_3_CR")
class LLaMa3_3Judge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.LLaMa3_3")()
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(LLaMa3_3Judge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')

@register_class(alias="Agent.Judge.Ministral8B_CR")
class Ministral8BJudge_criminalPrediction(Judge_criminalPrediction):
    def __init__(self, args, judge_info=None, name="B"):
        engine = registry.get_class("Engine.Ministral8B")()
        
        #编写profile
        id = judge_info['id']
        defendant_info = judge_info['defendant']
        court_information = judge_info['court_information']
        
        with open("./src/agents/profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
        if args.scenario == "J1Bench.Scenario.CR":
            profile = profiles['judge_CR']
            system_prompt = ''
            
            for p in profile:
                if '{defendant_name}' in p:
                    system_prompt += p.format(defendant_name = defendant_info['personal_information']['name']) + '\n'
                elif '{defendant_sex}' in p:
                    system_prompt += p.format(defendant_sex = defendant_info['personal_information']['sex']) + '\n'
                elif '{defendant_birth}' in p:
                    system_prompt += p.format(defendant_birth = defendant_info['personal_information']['birth_date']) + '\n'
                elif '{defendant_ethnicity}' in p:
                    system_prompt += p.format(defendant_ethnicity = defendant_info['personal_information']['ethnicity']) + '\n'
                elif '{defendant_address}' in p:
                    system_prompt += p.format(defendant_address = defendant_info['personal_information']['address']) + '\n'
                elif '{occupation}' in p:
                    system_prompt += p.format(occupation = defendant_info['personal_information']['occupation']) + '\n'
                elif '{education}' in p:
                    system_prompt += p.format(education = defendant_info['personal_information']['education']) + '\n'
                elif '{status}' in p:
                    system_prompt += p.format(status = defendant_info['personal_information']['status']) + '\n'
                elif '{third_party_findings}' in p:
                    if len(court_information['third_party_findings']) > 0:
                        system_prompt += p.format(third_party_findings = court_information['third_party_findings']) + '\n\n'
                    else:
                        system_prompt += p.format(third_party_findings = '无') + '\n\n'
                elif '{' not in p:
                    system_prompt += p + '\n'
            if system_prompt.endswith('\n'):
                system_prompt = system_prompt[:-1]
            self.system_prompt = system_prompt
            
        super(Ministral8BJudge_criminalPrediction, self).__init__(engine)
        self.judge_greetings = '现在开庭。'
        self.id = id
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--judge_temperature', type=float, default=0, help='temperature')
        parser.add_argument('--judge_max_tokens', type=int, default=4096, help='max tokens')
        parser.add_argument('--judge_top_p', type=float, default=1, help='top p')
        parser.add_argument('--judge_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--judge_presence_penalty', type=float, default=0, help='presence penalty')
    