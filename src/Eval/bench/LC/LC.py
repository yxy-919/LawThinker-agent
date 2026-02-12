import random
import os
import re
import sys
from collections import defaultdict
import os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(parent_dir)
import utils.utils_func as func
import argparse

class LCEvaluator:
    def __init__(self, args):
        self.ground_truth = func.load_jsonl('./src/data/case/J1-Eval_LC.jsonl')
        self.args = args
        
        self.dialog_history_dir = args.dialog_history_dir
        self.intermediate_eval = args.intermediate_eval
        self.final_eval = args.final_eval
    
    def chinese_to_arabic(self,chinese_num):
        digit_map = {
            '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
        }
        unit_map = {
            '十': 10, '百': 100, '千': 1000,
            '万': 10000, '亿': 100000000
        }
        result = 0      # 最终结果
        section = 0     # 当前“万/亿”以内的值
        number = 0      # 当前读到的数字
        i = 0
        while i < len(chinese_num):
            char = chinese_num[i]
            if char in digit_map:
                number = digit_map[char]
                i += 1

            elif char in unit_map:
                unit_value = unit_map[char]

                # 处理 十/百/千
                if unit_value < 10000:
                    # 关键修正：如“十七”，十前面没有数字，默认是 1
                    if number == 0:
                        number = 1

                    section += number * unit_value
                    number = 0

                # 处理 万/亿
                else:
                    section = (section + number) * unit_value
                    result += section
                    section = 0
                    number = 0

                i += 1

            else:
                # 忽略非法字符
                i += 1

        return result + section + number
    
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument("--dialog_history_dir", default = "./src/data/dialog_history", type = str)
        parser.add_argument("--intermediate_eval", default = "./src/Eval/eval_result", type = str)
        parser.add_argument("--final_eval", default = "./src/Eval/final_result/LC", type = str)
    
    def map_dialog_question(self, questions):
        mapped_questions = {}
        for q in questions[1:]:
            q_index = questions.index(q)
            turn = q['turn']
            former_question = q['former_question'].replace('?','').replace('？','').replace('“','').replace('”','').replace('话题：','').replace('它的类型是：','').replace('问答题','').replace('法律题','').replace('判断题','')
            former_answer = q['former_answer']
            if former_question == '无':
                continue
            if former_question not in mapped_questions:
                mapped_questions[former_question] = {
                    'turn': [turn],
                    'answer': [former_answer]
                }
            else:
                mapped_questions[former_question]['turn'].append(turn)
                mapped_questions[former_question]['answer'].append(former_answer)
                
        return mapped_questions
    
    def evaluate(self, ground_truth, model_questions, dialog_history):
        judgement_scores = 0
        binary_scores = 0
        nonbinary_scores = 0
        total_binary_scores = 0
        total_nonbinary_scores = 0
        total_scores = 0
        
        ground_truth_qa_ls = ground_truth['topic_list']
        
        def remove_article_items(text):
            # 匹配“第x款”，x 可以是中文数字或阿拉伯数字
            pattern = r"第[一二三四五六七八九十百千万\d]+款"
            return re.sub(pattern, '', text)
        
        skip_count = 0
        check_ls = []
        for q in ground_truth_qa_ls:
            question = q['topic']
            ground_truth_answer = q['content']
            type = q['type']
            
            flag = False
            for mqk in list(model_questions.keys()):
                a = mqk.replace('?','').replace('？','').replace('“','').replace('”','').replace('话题：','').replace('它的类型是：','').replace('问答题','').replace('法律题','').replace('判断题','')
                b = question.replace('?','').replace('？','').replace('“','').replace('”','')
                if a in b or b in a:
                    flag = True
                    break
                    
            if flag:
                relevant_turns = model_questions[mqk]['turn']
                temp = []
                for relevant_turn in relevant_turns:
                    for dh in dialog_history:
                        if dh['turn'] == relevant_turn + 1 and dh['role'] == 'Legal Trainee':
                            temp.append(dh['content'])
                            break
                model_answers = '\n'.join(temp)
            else:
                skip_count += 1
                continue
            
            # 答案是二分类
            if type == "binary":
                model_answer_binary = []
                for ma in model_questions[mqk]['answer']:
                    if ma.startswith('是'):
                        model_answer_binary.append(ma[0])
                    elif ma.startswith('否'):
                        model_answer_binary.append(ma[0])
                    else:
                        model_answer_binary.append(ma)
                        continue
                    
                if all(mba == "是" for mba in model_answer_binary) or all(mba == "否" for mba in model_answer_binary):
                    if model_answer_binary[0] == ground_truth_answer:
                        score = 1
                        binary_scores += 1
                        total_binary_scores += 1
                    else:
                        score = 0
                        binary_scores += 0
                        total_binary_scores += 1
                # 如果所有的都表示不一定
                elif all(len(mba) > 1 for mba in model_answer_binary):
                    score = 0.5
                    binary_scores += 0.5
                    total_binary_scores += 1
                # 剩余情况（混杂）
                else:
                    score = 0
                    binary_scores += 0
                    total_binary_scores += 1
                
                
                check_ls.append({
                    'question': question,
                    'model_answer': model_answer_binary,
                    'ground_truth_answer': ground_truth_answer,
                    'score': score
                })
                continue
            
            # 答案是法律
            elif type == "open-ended_law":
                law_pattern = re.compile(r'《.*?》')
                if re.search(law_pattern, ground_truth_answer):
                    law = re.search(law_pattern, ground_truth_answer).group(0).replace('《','').replace('》','') # 道路交通安全法
                    rest = ground_truth_answer.replace(law, '').replace('《','').replace('》','') # 第十七条
                    rest2 = '第' + str(self.chinese_to_arabic(rest)) + '条' # 第17条
                    rest = re.sub(r'第[一二三四五六七八九十百千0-9]+款', '', rest) # 
                    rest2 = re.sub(r'第[一二三四五六七八九十百千0-9]+款', '', rest2)
                        
                if law in model_answers and (rest in model_answers or rest2 in model_answers):
                    score = 1
                    nonbinary_scores += 1
                else:
                    score = 0
                total_nonbinary_scores += 1
                    
                check_ls.append({
                    'question': question,
                    'model_answer': model_answers,
                    'ground_truth': ground_truth_answer,
                    'score': score
                })
                    
            # 答案是其他
            else:
                prompt_sen = '''你是一名法律专家。请根据“法律问题”和“标准答案”，判断“待评测答案”是否完全、准确地涵盖了“标准答案”的所有核心要点，由此给出0-10分的评分。你**不用考虑待评测答案表达是否简洁、重点是否突出、是否使用寒暄语、结构是否冗长等非实质性因素**，无须因不够简洁而扣分。
                
                法律问题：
                {gt_question}

                标准答案：
                {gt_answer}

                待测评答案：
                {model_answer}
                
                以如下格式输出你的结果（中文括号分割，不要换行，不要带括号）：
                评分：；原因：
                '''
                full_prompt_sen = prompt_sen.format(gt_question = question, gt_answer = ground_truth_answer, model_answer = model_answers)
                # 修改
                sen_eval = func.get_completion(full_prompt_sen, [], flag=0)[0]
                sen_score = float(sen_eval.split('；')[0].replace('评分：','').replace('（','').replace('）',''))/10
                reason = sen_eval.split('；')[1]
                nonbinary_scores += sen_score
                total_nonbinary_scores += 1
                check_ls.append({
                    'question': question,
                    'model_answer': model_answers,
                    'ground_truth': ground_truth_answer,
                    'score': sen_score
                })
                continue
            
        result = {
            'BIN': binary_scores/total_binary_scores if total_binary_scores > 0 else 0,
            'NBIN': nonbinary_scores/total_nonbinary_scores if total_nonbinary_scores > 0 else 0,
            'check_ls': check_ls
        }
        return result
    
    def get_question(self, model, case):
        questions = case['check_mechanism']
        mapped_questions = self.map_dialog_question(questions)
        dialog_history = case['dialog_history']
        case_id = int(case['case_id'].split('-')[-1])
        for gt in self.ground_truth:
            if int(gt['id'].split('-')[-1]) == case_id:
                break
        result = self.evaluate(gt, mapped_questions, dialog_history)
        func.save_json(result, os.path.join(self.intermediate_eval, model, 'LC', f'LC_{case_id}.json'))
    
    def get_final_score(self, intermediate_model_folder, model):
        # 算总分
        binary_scores = []
        nonbinary_scores = []
        for file_name in os.listdir(intermediate_model_folder):
            data = func.load_json(os.path.join(intermediate_model_folder, file_name))
            binary_acc = data['BIN']
            nonbinary_acc = data['NBIN']
            binary_scores.append(binary_acc)
            nonbinary_scores.append(nonbinary_acc)
        print('binary_scores: ', np.mean(binary_scores))
        print('nonbinary_scores: ', np.mean(nonbinary_scores))
        result = {
            'model': model,
            'BIN': np.mean(binary_scores),
            'NBIN': np.mean(nonbinary_scores),
            'AVE': np.mean([np.mean(binary_scores), np.mean(nonbinary_scores)])
        }
        func.save_json(result, os.path.join(self.final_eval, f'{model}_final.json'))

    
    def iterate(self):
        for model in os.listdir(self.dialog_history_dir):
            print(f'Processing {model}')
            processed_ids = []
            intermediate_model_folder = os.path.join(self.intermediate_eval, model, 'LC')
            if not os.path.exists(intermediate_model_folder):
                os.makedirs(intermediate_model_folder)
            
            for file_name in os.listdir(intermediate_model_folder):
                case_id = file_name.split('_')[1].split('.')[0]
                processed_ids.append(int(case_id))
            
            self.dialog_history = func.load_jsonl(os.path.join(self.dialog_history_dir, model,'LC_dialog_history.jsonl'))
            
            for case in self.dialog_history:
                id = case['case_id']
                print(f'Processing case {id}')
                
                flag = True
                lawd = [d['content'] for d in case['dialog_history'] if d['role'] == 'Lawyer']
                for ld in lawd:
                    if ld is None:
                        result = {
                            'model': model,
                            'binary_scores': 0,
                            'nonbinary_scores': 0,
                            'ave': 0
                        }
                        func.save_json(result, os.path.join(intermediate_model_folder, f'LC_{id}.json'))
                        flag = False
                        break
                
                if int(id.split('-')[-1]) in processed_ids:
                    print(f'Case {id} has been processed!')
                    continue
                
                if flag:
                    self.get_question(model,case)
            self.get_final_score(intermediate_model_folder, model)

def main():
    parser = argparse.ArgumentParser()
    LCEvaluator.add_parser_args(parser)
    args = parser.parse_args()
    evaluator = LCEvaluator(args)
    evaluator.iterate()

if __name__ == '__main__':
    main()