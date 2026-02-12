import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import json
from typing import List
import jsonlines
from tqdm import tqdm
import threading
import time
import concurrent
import random
from utils.register import register_class, registry
from utils.utils_func import load_jsonl
import asyncio


@register_class(alias='J1Bench.Scenario.CR')
class CR:
    def __init__(self, args):
        case_database = load_jsonl(args.case_database)
        self.args = args
        self.case_quadric = []
        for case in case_database:
            defendant = registry.get_class(args.defendant)(
                args,
                defendant_info = case
                )
            lawyer = registry.get_class(args.lawyer)(
                args,
                lawyer_info = case
                )
            procurator = registry.get_class(args.procurator)(
                args,
                procurator_info = case
                )
            judge = registry.get_class(args.judge)(
                args,
                judge_info = case
                )
            defendant.id = case['id']
            lawyer.id = case['id']
            procurator.id = case['id']
            judge.id = case['id']
            self.case_quadric.append((defendant, lawyer, procurator, judge))
            
        self.max_conversation_turn = args.max_conversation_turn
        self.save_path = args.save_path
        self.max_workers = args.max_workers
        self.lock = threading.Lock()
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument("--case_database", default = "./src/data/case/J1-Eval_CR.jsonl", type=str)
        parser.add_argument("--defendant", default="Agent.Defendant.GPT_CR", help="registry name of defendant agent")
        parser.add_argument("--lawyer", default="Agent.Lawyer.GPT_CR", help="registry name of lawyer agent")
        parser.add_argument("--procurator", default="Agent.Procurator.GPT_CR", help="registry name of Procurator agent")
        parser.add_argument("--judge", default="Agent.Judge.GPT_CR", help="registry name of judge agent")
        parser.add_argument("--max_conversation_turn", default=50, type=int, help="max conversation turn")
        parser.add_argument("--save_path", default="./src/data/dialog_history/32b/CR_dialog_history.jsonl", help="save path for dialog history")
        parser.add_argument("--max_workers", default=100, type=int, help="max workers for parallel diagnosis")
        
    def remove_processed_cases(self): # 移除掉已经处理过的cases
        processed_case_ids = {}
        if os.path.exists(self.save_path):
            with jsonlines.open(self.save_path, "r") as f:
                for obj in f:
                    processed_case_ids[obj["case_id"]] = 1
            f.close()
        client_num = len(self.case_quadric)
        for i, client in enumerate(self.case_quadric[::-1]): #从头到尾返回一个新的、元素顺序相反的序列
            print(client[0].id)
            if processed_case_ids.get(client[0].id) is not None:
                self.case_quadric.pop((client_num-(i+1)))
            
        # random.shuffle(self.case_quadric)
        print("To-be-consulted case Number: ", len(self.case_quadric))
        
    def run(self):
        self.remove_processed_cases()
        for quadric in tqdm(self.case_quadric):
            defendant = quadric[0]
            lawyer = quadric[1]
            procurator = quadric[2]
            judge = quadric[3]
            self._criminal_prediction(defendant, lawyer, procurator, judge)
    
    async def parallel_run(self):
        self.remove_processed_cases()

        st = time.time() 
        print("Parallel Consult Start")
        semaphore = asyncio.Semaphore(self.max_workers)
        tasks = []
        for defendant, lawyer, procurator, judge in self.case_quadric:
            tasks.append(self._criminal_prediction(defendant, lawyer, procurator, judge, semaphore))
        with tqdm(total=len(tasks)) as pbar:
            async def track_progress(task):
                result = await task
                pbar.update(1)
                return result   
            tracked_tasks = [track_progress(task) for task in tasks]
            results = await asyncio.gather(*tracked_tasks)

        print("duration: ", time.time() - st)
    
    def save_dialog_info(self, dialog_info):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with jsonlines.open(self.save_path, "a") as f:
            f.write(dialog_info)
                
    async def _criminal_prediction(self, defendant, lawyer, procurator, judge, semaphore):
        dialog_history = [{"turn": 0, "role": "Judge", "content": judge.judge_greetings}]
        print("############### Dialog ###############")
        print("--------------------------------------")
        print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
        print(dialog_history[-1]["content"])
        
        flag = True
        for turn in range(self.max_conversation_turn):
            judge_response, judge_reasoning = await judge.speak_with_thinker(dialog_history[-1]["content"], turn, semaphore)
            if judge_response is None:
                break
            judge_response = judge_response.replace('</s>','')
            dialog_history.append({"turn": turn+1, "role": "Judge", "content": judge_response, "reasoning": judge_reasoning})
            print("--------------------------------------")
            print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
            print(dialog_history[-1]["content"])
            
                
            if '对公诉机关说' in judge_response:
                prosecution_response = await procurator.speak(judge_response.replace('对公诉机关说：', '').replace('对被告当事人说', '').replace('对被告辩护人说：', '').replace('（对被告当事人说）',''), semaphore)
                dialog_history.append({"turn": turn+1, "role": "Procurator", "content": prosecution_response})
            elif '对被告当事人说' in judge_response:
                defendant_response = await defendant.speak(judge_response.replace('对公诉机关说：', '').replace('对被告当事人说', '').replace('对被告辩护人说：', '').replace('（对被告当事人说）',''), semaphore)
                dialog_history.append({"turn": turn+1, "role": "Defendant", "content": defendant_response})
                if defendant_response == "RateLimitError":
                    flag = False
                    break
            elif '对被告辩护人说' in judge_response:
                lawyer_response = await lawyer.speak(judge_response.replace('对公诉机关说：', '').replace('对被告当事人说', '').replace('对被告辩护人说：', '').replace('（对被告当事人说）',''), semaphore)
                dialog_history.append({"turn": turn+1, "role": "Lawyer", "content": lawyer_response})
                
            print("--------------------------------------")
            print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
            print(dialog_history[-1]["content"])
            
            if '结束庭审' in judge_response or 'Failed to generate response' in judge_response:
                break
        
        if flag:
            dialog_info = {
                "case_id": defendant.id,
                "judge": self.args.judge,
                "judge_engine_name": judge.engine.model_path,
                "procurator": self.args.procurator,
                "procurator_engine_name": procurator.engine.model_path,
                "lawyer": self.args.lawyer,
                "lawyer_engine_name": lawyer.engine.model_path,
                "defendant": self.args.defendant,
                "defendant_engine_name": defendant.engine.model_path,
                "dialog_history": dialog_history
            }
            print(dialog_info)
            self.save_dialog_info(dialog_info)