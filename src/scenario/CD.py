import argparse
import os
import json
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from typing import List
import jsonlines
from tqdm import tqdm
import time
import concurrent
import random
from utils.register import register_class, registry
from openai import OpenAI
from requests.exceptions import ConnectionError, Timeout, RequestException
import threading
from utils.utils_func import load_jsonl
import asyncio

@register_class(alias='J1Bench.Scenario.CD')
class CD:
    def __init__(self, args):
        case_database = load_jsonl(args.case_database)
        # case_database = case_database[:1]
        self.args = args
        self.case_pair = []
        for case in case_database:
            specific_character = registry.get_class(args.specific_character)(
                args,
                specific_character_info = case
                )
            lawyer = registry.get_class(args.lawyer)(
                args,
                lawyer_info = case
                )
            specific_character.id = case['id']
            lawyer.id = case['id']
            self.case_pair.append((specific_character, lawyer))
            
        self.max_conversation_turn = args.max_conversation_turn
        self.save_path = args.save_path
        self.max_workers = args.max_workers
        self.lock = threading.Lock()
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument("--case_database", default = "./src/data/case/J1-Eval_CD.jsonl", type=str)
        parser.add_argument("--specific_character", default="Agent.Specific_character.GPT_CD", help="registry name of specific character agent") 
        parser.add_argument("--lawyer", default="Agent.Lawyer.GPT_CD", help="registry name of lawyer agent")
        parser.add_argument("--max_conversation_turn", default=30, type=int, help="max conversation turn between the lawyer and the specific character")
        parser.add_argument("--save_path", default="./src/data/dialog_history/32b/CD_dialog_history.jsonl", help="save path for dialog history")
        parser.add_argument("--max_workers", default=100, type=int, help="max workers for parallel CD")
        
    def remove_processed_cases(self): # 移除掉已经处理过的cases
        processed_case_ids = {}
        if os.path.exists(self.save_path):
            with jsonlines.open(self.save_path, "r") as f:
                for obj in f:
                    processed_case_ids[obj["case_id"]] = 1
            f.close()
        specific_character_num = len(self.case_pair)
        for i, specific_character in enumerate(self.case_pair[::-1]): #从头到尾返回一个新的、元素顺序相反的序列
            print(specific_character[0].id)
            if processed_case_ids.get(specific_character[0].id) is not None:
                self.case_pair.pop((specific_character_num-(i+1)))
            
        # random.shuffle(self.case_pair)
        print("To-be-consulted case Number: ", len(self.case_pair))

    def run(self):
        self.remove_processed_cases()
        for pair in tqdm(self.case_pair):
            specific_character = pair[0]
            lawyer = pair[1]
            self._consult(specific_character, lawyer)
    
    async def parallel_run(self):
        self.remove_processed_cases()

        st = time.time() 
        print("Parallel Consult Start")
        semaphore = asyncio.Semaphore(self.max_workers)
        tasks = []
        for specific_character, lawyer in self.case_pair:   
            tasks.append(self._consult(specific_character, lawyer, semaphore))
        with tqdm(total=len(tasks)) as pbar:
            async def track_progress(task):
                result = await task
                pbar.update(1)
                return result
            tracked_tasks = [track_progress(task) for task in tasks]
            results = await asyncio.gather(*tracked_tasks)
        print("duration: ", time.time() - st)
    
    async def _consult(self, specific_character, lawyer, semaphore):
        dialog_history = [{"turn": 0, "role": "Specific Character", "content": specific_character.specific_character_greetings}]
        print("############### Dialog ###############")
        print("--------------------------------------")
        print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
        print(dialog_history[-1]["content"])
        
        
        for turn in range(self.max_conversation_turn):
            lawyer_response = await lawyer.speak_with_thinker(dialog_history[-1]["content"], specific_character.id, turn, semaphore)
            if lawyer_response is None:
                break
            lawyer_response_text = lawyer_response[0].replace('<think>','').replace('</think>','').replace('</s>','')
            lawyer_reasoning = lawyer_response[1]
            dialog_history.append({"turn": turn+1, "role": "Lawyer", "content": lawyer_response_text, "reasoning": lawyer_reasoning})
            
            # lawyer_response = await lawyer.speak(dialog_history[-1]["content"], specific_character.id, semaphore)
            # lawyer_response_text = lawyer_response.replace('<think>','').replace('</think>','').replace('</s>','')
            # dialog_history.append({"turn": turn+1, "role": "Lawyer", "content": lawyer_response_text})
            
            print("--------------------------------------")
            print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
            print(dialog_history[-1]["content"])
            if '询问结束' in lawyer_response_text:
                break
            
            specific_character_response = await specific_character.speak(lawyer_response_text, semaphore)
            dialog_history.append({"turn": turn+1, "role": "Specific Character", "content": specific_character_response})
            
            
            print("--------------------------------------")
            print(dialog_history[-1]["turn"], dialog_history[-1]["role"])
            print(dialog_history[-1]["content"])
            
        dialog_info = {
            "case_id": specific_character.id,
            "lawyer": self.args.lawyer,
            "lawyer_engine_name": lawyer.engine.model_path,
            "specific_character": self.args.specific_character,
            "specific_character_engine_name": specific_character.engine.model_path,
            "dialog_history": dialog_history
        }
        print(dialog_info)
        self.save_dialog_info(dialog_info)
        
    
    def save_dialog_info(self, dialog_info):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with jsonlines.open(self.save_path, "a") as f:
            f.write(dialog_info)