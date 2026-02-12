
import argparse
import os
from openai import OpenAI
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.utils_func import get_completion_extract
import json
from typing import List
import threading
import jsonlines
from tqdm import tqdm
import time
import concurrent
import random
from utils.register import register_class, registry
from utils.utils_func import load_jsonl
import re
import asyncio



@register_class(alias='J1Bench.Scenario.LC')
class LC:
    def __init__(self, args):
        case_database = load_jsonl(args.case_database)
        self.args = args
        self.case_pair = []
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
        parser.add_argument("--case_database", default = "./src/data/case/J1-Eval_LC.jsonl", type=str)
        parser.add_argument("--general_public", default="Agent.General_public.LC_GPT", help="registry name of general_public agent") 
        parser.add_argument("--trainee", default="Agent.Trainee.LC_GPT", help="registry name of trainee agent")
        parser.add_argument("--max_conversation_turn", default=10, type=int, help="max conversation turn between the trainee and the general public")
        parser.add_argument("--save_path", default="./src/data/dialog_history/32b/LC_dialog_history.jsonl", help="save path for dialog history")
        parser.add_argument("--max_workers", default=100, type=int, help="max workers for parallel LC")
        
    def remove_processed_cases(self):
        processed_case_ids = {}
        if os.path.exists(self.save_path):
            with jsonlines.open(self.save_path, "r") as f:
                for obj in f:
                    processed_case_ids[obj["case_id"]] = 1
            f.close()
        general_public_num = len(self.case_pair)
        for i, general_public in enumerate(self.case_pair[::-1]):
            print(general_public[0].id)
            if processed_case_ids.get(general_public[0].id) is not None:
                self.case_pair.pop((general_public_num-(i+1)))
            
        # random.shuffle(self.case_pair)
        print("To-be-consulted case Number: ", len(self.case_pair))
        
    def run(self):
        self.remove_processed_cases()
        for pair in tqdm(self.case_pair):
            general_public = pair[0]
            trainee = pair[1]
            dialog_info = self._consult(general_public, trainee)
            # questions_log = dialog_info['check_mechanism']
            # unquestioned_ls = [ql['unquestioned'] for ql in questions_log]
            # if '无' in unquestioned_ls:
            #     self.save_dialog_info(dialog_info)
    
    async def parallel_run(self):
        self.remove_processed_cases()

        st = time.time() 
        print("Parallel Consult Start")
        semaphore = asyncio.Semaphore(self.max_workers)
        max_tool_calls = 100
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
        dialog_history = [{"turn": 0, "role": "General Public", "content": general_public.general_public_greetings}]
        print("############### Dialog ###############")
        print("--------------------------------------")
        print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
        print(dialog_history[-1]["content"])
        
        topic_list = general_public.topic_list
        notasked_questions = general_public.topic_list
        next_action = '无'
        
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
            questions += f'问题：{question}？它的类型是：{new_type}\n'
        questions = questions.replace('？？', '？').replace('?？','？')
        unquestioned = questions
        
        role = general_public.role
        former_answer = '无'
        former_question = '无'
        points_of_confusion = []
        for turn in range(self.max_conversation_turn):
            if '结束对话' in next_action:
                break

            trainee_response, trainee_reasoning = await trainee.speak_with_thinker(dialog_history[-1]['content'], general_public.id, semaphore, tool_calls_semaphore)
            trainee_response = trainee_response.replace('<think>','').replace('</think>','')
            dialog_history.append({"turn": turn+1, "role": "Legal Trainee", "content": trainee_response, "reasoning": trainee_reasoning})

            # trainee_response = await trainee.speak(dialog_history[-1]['content'], general_public.id, semaphore)
            # if trainee_response == None:
            #     break
            # dialog_history.append({"turn": turn+1, "role": "Legal Trainee", "content": trainee_response})
            
            dialogue = ''
            for d in dialog_history[-2:]:
                if d['role'] == 'Legal Trainee':
                    content = d['content']
                    dialogue += f'法律顾问: {content}\n'
                if d['role'] == 'General Public':
                    content = d['content']
                    dialogue += f'{role}: {content}\n'
            
            
            memory_prompt = '''你是一起事件的当事人助手，根据当前对话状态和事件经过，完成以下两个任务。
            
            ## 任务1：确定“这一轮咨询问题”
            根据当前对话内容的当事人发言，从参考问题列表中选取一个最符合的问题输出。如果无法匹配到，则输出“无”。
            
            ## 任务2：确定“这一轮问题的答案”
            结合律师在当前对话内容中的发言，以及“这一轮咨询的问题”，总结律师对该问题的回答。具体规则如下：
            （一）如果“这一轮的咨询话题”与“之前的咨询话题”一致，确保“这一轮的话题答案”与之前话题答案完全一致。不允许出现前后矛盾的情况。
            （二）若为判断题（是/否类问题），则只考虑最普通的情况，直接回答“是”或“否”，不能回答除“是”或“否”以外的内容。
            （三）若为一般性问题，则严格总结律师的核心答复，确定的“这一轮的答案”中不要出现分号（；或;）。】
            （四）若“这一轮咨询的问题”为“无”，则“这一轮问题的答案”也为“无”。若律师没有解答这一问题，则“这一轮的答案”也为“无”，不要自行解答疑问。
            
            ##当前对话状态
            参考问题列表：
            {questions}
            
            之前的咨询话题及答案：
            {previous_questions_and_answers}
            
            当前对话内容：
            {dialog_history}
    
            ## 最终输出格式（请严格使用英文分号隔开，不换行）：
            这一轮咨询问题：;这一轮问题的答案：
            '''
            ex_questions = ''
            count = 1
            for q in points_of_confusion:
                question = q['former_question']
                if question not in ex_questions and question.replace('。','') != '无':
                    ex_questions += f'{count}. {question}\n'
                    count += 1
            if ex_questions == '':
                ex_questions = '无'
            if ex_questions.endswith('\n'):
                ex_questions = ex_questions[:-1]
            
            # 拼接之前的咨询话题及答案
            previous_questions_and_answers = ''
            for q in points_of_confusion:
                question = q['former_question']
                answer = q['former_answer']
                if question != '无' and answer != '无':
                    previous_questions_and_answers += f'咨询的话题：{question}\n答案：{answer}\n'
            if previous_questions_and_answers == '':
                previous_questions_and_answers = '无'
            
            full_prompt = memory_prompt.format(
                dialog_history = dialogue, 
                questions = questions,
                previous_questions_and_answers = previous_questions_and_answers
                )  
            full_prompt = full_prompt.replace(' ','').replace('\n\n\n\n','\n\n').replace('\n\n\n', '\n\n')
            if full_prompt.endswith('\n'):
                full_prompt = full_prompt[:-1]
            message = [
                {'role': 'system', 'content': f'你是{role}'},
                {'role': 'user', 'content': full_prompt}
            ]
            memory_response = (await get_completion_extract(tool_calls_semaphore, full_prompt, [], flag=0))[0]
            former_question = memory_response.split(';')[0].replace('这一轮咨询问题：', '')
            former_answer = memory_response.split(';')[-1].replace('这一轮问题的答案：', '')
            
            
            # 确定没有问过的问题
            if former_question != '无':
                notasked_questions = [notq for notq in notasked_questions if former_question.replace('话题：','').replace('”','').replace('“','').replace('问答题','').replace('判断题','').replace('法律题','').replace('它的类型是：','').replace('?','').replace('？', '') not in notq['topic'].replace('?','').replace('？','').replace('”','').replace('“','')]
                unquestioned = ''
                count = 1
                if len(notasked_questions) > 0:
                    for nq in notasked_questions:
                        temp_q = nq['topic']
                        unquestioned += f'问题{count}. {temp_q}\n'
                        count += 1
                else:
                    unquestioned += '无'
                if unquestioned.endswith('\n'):
                    unquestioned = unquestioned[:-1]
            
            temp = re.search(r'(以下是还没有完成咨询的问题列表：\n.*(?:\n.*)*)', general_public.memories[0][1]).group(1).strip()
            memory_module = f'以下是还没有完成咨询的问题列表：\n{unquestioned}'
            new_profile = general_public.memories[0][1].replace(temp, memory_module).replace('\n\n\n','\n\n')
            if new_profile.endswith('\n'):
                new_profile = new_profile[:-1]
            system_prompt = list(general_public.memories[0])
            system_prompt[1] = new_profile
            general_public.memories[0] = tuple(system_prompt)
            
            
            points_of_confusion.append({
                'turn': turn,
                'former_question': former_question, 
                'former_answer': former_answer,
                'unquestioned': unquestioned
            })
            
            
            print("--------------------------------------")
            print(f'{turn+1}')
            print(f'新一轮问题的答案：{former_answer}，新一轮咨询问题：{former_question}')
            
            count = 1
            # 修改
            general_public_response = await general_public.speak(trainee_response, semaphore)
            dialog_history.append({"turn": turn+1, "role": "General Public", "content": general_public_response})
            
            print("--------------------------------------")
            print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
            print(dialog_history[-1]["content"])
            
            if '结束对话' in general_public_response:
                break
          
        # 统一判断题答案
        # 按问题聚合所有轮次的回答
        from collections import defaultdict
        question_groups = defaultdict(list)
        for idx, itm in enumerate(points_of_confusion):
            question_groups[itm['former_question']].append((idx, itm))

        def is_judge_answer(ans: str) -> bool:
            ans = ans.strip()
            return ans.startswith('是') or ans.startswith('否')

        for q, records in question_groups.items():
            # 跳过非判断题（没有任何以 是/否 开头的回答）
            has_judge = any(is_judge_answer(r[1]['former_answer']) for r in records)
            if not has_judge:
                continue
            # 找到第一条格式正确的回答作为标准
            std_answer = None
            for _, rec in records:
                if is_judge_answer(rec['former_answer']):
                    std_answer = rec['former_answer']
                    break
            if std_answer is None:
                continue
            # 统一该问题下所有记录的答案
            for _, rec in records:
                if rec['former_answer'] != std_answer:
                    rec['former_answer'] = std_answer

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
        # return dialog_info
        questions_log = dialog_info['check_mechanism']
        unquestioned_ls = [ql['unquestioned'] for ql in questions_log]
        # 修改
        # if '无' in unquestioned_ls:
        self.save_dialog_info(dialog_info)
    
    def save_dialog_info(self, dialog_info):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with jsonlines.open(self.save_path, "a") as f:
            f.write(dialog_info)