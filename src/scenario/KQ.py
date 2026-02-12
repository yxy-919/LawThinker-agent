
import argparse
import os
import sys
import json
from openai import OpenAI
from requests.exceptions import ConnectionError, Timeout, RequestException

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.utils_func import get_completion_extract
import threading
import re
import json
from typing import List
import jsonlines
from tqdm import tqdm
import time
import concurrent
import random
from utils.register import register_class, registry
from utils.utils_func import load_jsonl
import asyncio

@register_class(alias='J1Bench.Scenario.KQ')
class KQ:
    def __init__(self, args):
        case_database = load_jsonl(args.case_database)
        # case_database = case_database[:2]
        self.args = args
        self.case_pair = []
        
        # 如果想修改跑的项，修改此处即可
        for case in case_database:
            general_public = registry.get_class(args.general_public)(
                args,
                general_public_info = case
                )
            trainee = registry.get_class(args.trainee)(
                args,
                trainee_info = case
                )
            general_public.id = case['id']
            trainee.id = case['id']
            self.case_pair.append((general_public, trainee))
            
        self.max_conversation_turn = args.max_conversation_turn
        self.save_path = args.save_path
        self.max_workers = args.max_workers
        self.lock = threading.Lock()
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument("--case_database", default = "./src/data/case/J1-Eval_KQ.jsonl", type=str)
        parser.add_argument("--general_public", default="Agent.General_public.ConsultGPT", help="registry name of general public agent") 
        parser.add_argument("--trainee", default="Agent.Trainee.ConsultGPT", help="registry name of trainee agent")
        parser.add_argument("--max_conversation_turn", default=15, type=int, help="max conversation turn between the trainee and the general public")
        parser.add_argument("--save_path", default="./src/data/dialog_history/32b/KQ_dialog_history.jsonl", help="save path for dialog history")
        parser.add_argument("--max_workers", default=100, type=int, help="max workers for parallel KQ")

    def remove_processed_cases(self):
        processed_case_ids = {}
        if os.path.exists(self.save_path):
            with jsonlines.open(self.save_path, "r") as f:
                for obj in f:
                    processed_case_ids[obj["case_id"]] = 1
            f.close()
        case_num = len(self.case_pair)
        for i, case in enumerate(self.case_pair[::-1]):
            # print(case[0].id)
            if processed_case_ids.get(case[0].id) is not None:
                self.case_pair.pop((case_num-(i+1))) #移除指定的索引
            
        # random.shuffle(self.case_pair)
        print("To-be-consulted case Number: ", len(self.case_pair))
        
    def run(self):
        self.remove_processed_cases()
        for pair in tqdm(self.case_pair):
            general_public = pair[0]
            trainee = pair[1]
            count = 1
            dialog_info = self._consult(general_public, trainee)
    
    async def parallel_run(self):
        self.remove_processed_cases()

        st = time.time() 
        print("Parallel Consult Start")
        semaphore = asyncio.Semaphore(self.max_workers)
        max_tool_calls = 5
        tool_calls_semaphore = asyncio.Semaphore(max_tool_calls)
        tasks = []
        for general_public, trainee in self.case_pair:
            tasks.append(self._consult(general_public, trainee, semaphore, tool_calls_semaphore))
        with tqdm(total=len(tasks)) as pbar:
            async def track_progress(task):
                result = await task
                pbar.update(1)
                return result
            tracked_tasks = [track_progress(task) for task in tasks]
            results = await asyncio.gather(*tracked_tasks)
        print("duration: ", time.time() - st)
    
    async def _consult(self, general_public, trainee, semaphore, tool_calls_semaphore):
        ## 启动对话
        dialog_history = [{"turn": 0, "role": "Legal Trainee", "content": trainee.trainee_greetings}]
        trainee.memorize(("assistant", trainee.trainee_greetings), general_public.id)
        print("############### Dialog ###############")
        print("--------------------------------------")
        print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
        print(dialog_history[-1]["content"])
        
        topic_list = general_public.topic_list
        unfinished_topics = general_public.topic_list
        legal_agent = general_public.legal_agent
        general_public_role = general_public.general_public_role
        
        points_of_confusion = []
        present_question = '无'
        present_answer = '无'
        new_profile = ''
        
        ## 记录问题列表
        questions = ''
        for q in topic_list:
            type = q['type']
            if type == 'qa':
                new_type = '问答题'
            elif type == 'law':
                new_type = '法律题'
            else:
                new_type = '判断题'
            question = q['topic']
            questions += f'话题：{question}？它的类型是：{new_type}\n'
        questions = questions.replace('？？', '？').replace('?？','？')
        if questions.endswith('\n'):
            questions = questions[:-1]
        unquestioned = questions
        
        
        ## 对话主体
        for turn in range(self.max_conversation_turn):
            # 大众回复
            general_public_response = await general_public.speak(dialog_history[-1]["content"], semaphore)
            dialog_history.append({"turn": turn+1, "role": "General Public", "content": general_public_response})
            
            print("--------------------------------------")
            print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
            print(dialog_history[-1]["content"])
            
            if '结束对话' in general_public_response:
                break
            
            # 法律智能体回复
            trainee_response, trainee_reasoning = await trainee.speak_with_thinker(general_public_response, general_public.id, semaphore, tool_calls_semaphore)
            trainee_response = trainee_response.replace('<think>','').replace('</think>','')
            dialog_history.append({"turn": turn+1, "role": "Legal Trainee", "content": trainee_response, "reasoning": trainee_reasoning})

            # trainee_response = await trainee.speak(general_public_response, general_public.id, semaphore)
            # dialog_history.append({"turn": turn+1, "role": "Legal Trainee", "content": trainee_response})
            
            print("--------------------------------------")
            print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
            print(dialog_history[-1]["content"])
            
            
            # 记忆库总结
            dialogue = ''
            for d in dialog_history[-2:]:
                if d['role'] == 'Legal Trainee':
                    content = d['content']
                    dialogue += f'{legal_agent}：{content}\n'
                if d['role'] == 'General Public':
                    content = d['content']
                    dialogue += f'{general_public_role}：{content}\n'
            if dialogue.endswith('\n'):
                dialogue = dialogue[:-1]
                
                
            
            memory_prompt = '''你是{general_public_role}的助手，根据当前对话状态，完成以下两个个任务。
            
            ##  任务1：确定“这一轮的咨询话题”
            根据当前对话内容的当事人发言，从参考问题列表中选取一个最符合的问题输出。如果无法匹配到，则输出“无”。
            
            ## 任务2：总结“这一轮的话题答案”
            依据{legal_agent}的当前对话内容与“这一轮的咨询话题”，总结其对该话题的回答：若为判断题（是/否类问题），则只考虑最普通的情况，直接回答“是”或“否”，不能回答除“是”或“否”以外的内容。若为一般性问题，则严格总结{legal_agent}的核心答复。若“这一轮的咨询话题”为“无”，则“话题答案”也为“无”。若律师没有解答这一问题，则“这一轮的答案”也为“无”，不要自行解答疑问。如果“这一轮的咨询话题”与“上一轮的咨询话题”一致，确保“这一轮的话题答案”与“上一轮的话题答案”一致。
            
            ## 当前对话状态
            参考话题列表：
            {questions}
            
            上一轮的咨询话题：
            {topic}
            
            上一轮的话题答案：
            {present_answer}
            
            当前对话内容：
            {dialog_history}

            ## 最终输出格式（请严格使用英文分号隔开，不换行）：
            这一轮的咨询话题：;这一轮的话题答案：
            '''
            ex_content = ''
            count = 1
            for q in points_of_confusion:
                question = q['question']
                if question not in ex_content and question != '无':
                    ex_content+=f'{count}. {question}\n'
                    count += 1
            if ex_content == '':
                ex_content = '无'
            if ex_content.endswith('\n'):
                ex_content = ex_content[:-1]
            
            # 主要从当前一轮的对话中提取出标准问题和答案，方便知道哪些问题已经被提问
            full_prompt = memory_prompt.format(
                questions = questions,
                dialog_history = dialogue, 
                legal_agent=legal_agent, 
                general_public_role = general_public_role, 
                topic = present_question,
                present_answer = present_answer
                )
            full_prompt = full_prompt.replace(' ','')
            if full_prompt.endswith('\n'):
                full_prompt = full_prompt[:-1]
            memory_response = (await get_completion_extract(tool_calls_semaphore, full_prompt, [], flag =0))[0]
            present_question = memory_response.split(';')[0].replace('这一轮的咨询话题：', '')
            present_answer = memory_response.split(';')[-1].replace('这一轮的话题答案：', '')
            
            # 确定没有问过的问题
            if present_question != '无':
                unfinished_topics = [notq for notq in unfinished_topics if present_question.replace('话题：','').replace('它的类型是：','').replace('?','').replace('？', '').replace('”','').replace('“','').replace('问答题','').replace('判断题','').replace('法律题','') not in notq['topic'].replace('?','').replace('？','').replace('”','').replace('“','')]
                unquestioned = ''
                count = 1
                if len(unfinished_topics) > 0:
                    for nq in unfinished_topics:
                        temp_q = nq['topic']
                        unquestioned += f'{count}. {temp_q}\n'
                        count += 1
                else:
                    unquestioned += '无'
                if unquestioned.endswith('\n'):
                    unquestioned = unquestioned[:-1]
            
            
            memory_module = f'这一轮的咨询的话题是：{present_question}。这一轮话题的答案是：{present_answer}'
            print("--------------------------------------")
            print(f'turn: {turn + 1}', memory_module)
            
            # 修改system prompt中还没被咨询的话题列表
            temp = re.search(r'(以下是还没有完成咨询的话题列表：\n.*(?:\n.*)*)', general_public.memories[0][1]).group(1).strip()
            memory_module = f'以下是还没有完成咨询的话题列表：\n{unquestioned}'
            new_profile = general_public.memories[0][1].replace(temp, memory_module)
            system_prompt = list(general_public.memories[0])
            system_prompt[1] = new_profile
            general_public.memories[0] = tuple(system_prompt)
            
            # check每一轮提的问题和答案
            points_of_confusion.append({
                'turn': turn,
                'question': present_question,
                'unquestioned': unquestioned, 
                'answer': present_answer
            })
        
        # 统一判断题答案
        # 按问题聚合所有轮次的回答
        from collections import defaultdict
        question_groups = defaultdict(list)
        for idx, itm in enumerate(points_of_confusion):
            question_groups[itm['question']].append((idx, itm))

        def is_judge_answer(ans: str) -> bool:
            ans = ans.strip()
            return ans.startswith('是') or ans.startswith('否')

        for q, records in question_groups.items():
            # 跳过非判断题（没有任何以 是/否 开头的回答）
            has_judge = any(is_judge_answer(r[1]['answer']) for r in records)
            if not has_judge:
                continue
            # 找到第一条格式正确的回答作为标准
            std_answer = None
            for _, rec in records:
                if is_judge_answer(rec['answer']):
                    std_answer = rec['answer']
                    break
            if std_answer is None:
                continue
            # 统一该问题下所有记录的答案
            for _, rec in records:
                if rec['answer'] != std_answer:
                    rec['answer'] = std_answer

            
        dialog_info = {
            "case_id": general_public.id,
            "trainee": self.args.trainee,
            "trainee_engine_name": trainee.engine.model_path,
            "general_public": self.args.general_public,
            "general_public_engine_name": general_public.engine.model_path,
            "dialog_history": dialog_history,
            "check_mechanism": points_of_confusion
        }
        print(dialog_info)
        questions_log = dialog_info['check_mechanism']
        unquestioned_ls = [ql['unquestioned'] for ql in questions_log]
        # 全部问题都被解决
        # 修改
        # if '无' in unquestioned_ls:
        self.save_dialog_info(dialog_info)
    
    def save_dialog_info(self, dialog_info):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with jsonlines.open(self.save_path, "a") as f:
            f.write(dialog_info)