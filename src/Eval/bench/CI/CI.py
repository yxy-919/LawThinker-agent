import re
import numpy as np
import os
from collections import defaultdict
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(parent_dir)
import utils.utils_func as func
import argparse


class CIEvaluator:
    def __init__(self, args):
        self.ground_truth = func.load_jsonl('./src/data/case/J1-Eval_CI.jsonl')
        self.args = args
        
        self.dialog_history_dir = args.dialog_history_dir
        self.intermediate_eval = args.intermediate_eval
        self.final_eval = args.final_eval
        
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument("--dialog_history_dir", default = "./src/data/dialog_history", type = str)
        parser.add_argument("--intermediate_eval", default = "./src/Eval/eval_result", type = str)
        parser.add_argument("--final_eval", default = "./src/Eval/final_result/CI", type = str)    
    
    
    def get_PFS(self, dialog_history):
        procedures = {
            '开庭审理': {
                '核实身份': {
                    'options': ['核实双方当事人的身份', '核实双方当事人身份', '核实当事人身份', '核实身份'],
                    'flag' : False
                },
                '审判庭构成': {
                    'options': ['构成', '合议庭', '本庭构成'],
                    'flag': False
                },
                '适用程序': {
                    'options': ['适用简易程序', '适用程序', '程序'],
                    'flag': False
                },
                '诉讼权利': {
                    'options': ['诉讼权利','告知双方当事人在诉讼中的权利'],
                    'flag' : False
                },
                '申请回避': {
                    'options': ['回避'],
                    'flag': False
                    },
                '缺席后果': {
                    'options': ['缺席审理', '缺席后果'],
                    'flag': False
                }
            },
            '法庭调查': {
                '原告陈述': {
                    'options': ['诉讼请求', '起诉事实'],
                    'flag': False
                },
                '被告答辩': {
                    'options': ['辩护', '答辩', '辩称'],
                    'flag': False
                },
                '举证质证': {
                    'options': ['出示证据', '质证'],
                    'flag': False
                },
                '法官发问': {
                    'options': ['发问'],
                    'flag': False
                },
            },
            '庭审辩论': {
                '争议焦点': {
                    'options': ['争议焦点'],
                    'flag': False
                },
                '最后陈述': {
                    'options': ['其他想要说', '其他要说', '其他陈述', '陈述'],
                    'flag': False
                }
            },
            '法庭调解': {
                '询问是否同意调解': {
                    'options': ['调解'],
                    'flag': False
                }
            }
        }
        
        UNI = 0
        
        ACT = 0
        judge_dialog = [dh for dh in dialog_history if dh['role'] == 'Judge']
        # 检查是否在一次对话中即出现多个阶段
        for jd in dialog_history:
            content = jd['content']
            role = jd['role']
            if role == 'Judge':
                # 对话成立性
                count = 0
                for p in ['<开庭审理>', '<法庭调查>', '<庭审辩论>', '<法庭调解>']:
                    if p in content:
                        count += 1
                    if count == 3:
                        PFS = {
                            'UNI': 0,
                            'STA': 0,
                            'ACT': 0
                        }
                        return PFS
        
        STA = 0
        scored_procedures = set()
        for idx, jd in enumerate(dialog_history):
            content = jd['content']
            role = jd['role']
            if role == 'Judge':
                # 程序正确性
                UNI = 1
                # 每个阶段的关键词是否存在
                if idx < len(dialog_history) - 1:
                    next_turn = dialog_history[idx + 1]
                    if next_turn['role'] == "Plaintiff's Lawyer" or next_turn['role'] == "Defendant's Lawyer":
                        for k in list(procedures.keys()):
                            actions = procedures[k]
                            for sk in list(actions.keys()):
                                options = procedures[k][sk]['options']
                                flag = actions[sk]['flag']
                                if not flag:
                                    for o in options:
                                        if o in content:
                                            ACT += 1
                                            procedures[k][sk]['flag'] = True
                                            break
                                            
                # STA衡量每个阶段关键词是否全部存在
                for k in procedures.keys():
                    if k in scored_procedures:
                        continue
                    procedure_flag = True
                    for sk in procedures[k].keys():
                        f = procedures[k][sk]['flag']
                        if not f:
                            procedure_flag = False
                            break
                    if procedure_flag:
                        STA += 1
                        scored_procedures.add(k)
        
        total_ACT = 0
        for tk in procedures.keys():
            total_ACT += len(list(procedures[tk].keys()))
        
        PFS = {
            'UNI': UNI, # 早退
            'STA': STA/4, # 每个阶段的关键词是否完整
            'ACT': ACT/total_ACT, # 所有阶段的关键词
            'check': procedures
        }
        return PFS
    
    def get_JUD_REA_LAW(self, model_judgment, gt_case):
        judgment_patterns = [
            r'判决：(.*?)<结束庭审>',
            r'判决：(.*?)法条',
            r'判决：(.*?)依据',
            r'本院判决(.*?)依据：',
            r'判决如下：(.*?)结束庭审',
            r'判决如下(.*?)审判长',
            r'判决结果(.*?)审判长',
            r'判决结果(.*?)结束庭审',
            r'判决结果(.*?)依据',
            r'判决结果(.*?)法条',
            r'判决如下：(.*?)。',
            r'判决结果(.*?)本案将择期宣判。',
            r"协议：(.*?)结束庭审",
            r"双方已自愿(.*?)本案依据",
            r"判令(.*?)判决依据",
            r'判决：(.*?)《',
            r'判决(.*?)依据',
            r'判决(.*?)结束庭审'
        ]
        for pattern in judgment_patterns:
            match = re.search(pattern, model_judgment.replace('*',''), re.DOTALL)
            if match:
                m_judgement = match.group(1).strip()
                break
        if match is None:
            m_judgement = model_judgment
        if m_judgement.startswith('\n'):
            m_judgement = m_judgement[1:]
        
            
        reason_patterns = [
            r'本院认为(.*?)判决',
            r'本院认为(.*?)依照',
            r'本院认为(.*?)依据'
            ]
        
        for pattern in reason_patterns:
            matches = list(re.finditer(pattern, model_judgment.replace('*',''), re.DOTALL))
            if matches:
                reasons = matches[-1].group(1).strip()
                break
        try:
            print(reasons)
        except:
            reasons = model_judgment
        
        # 判决准确性
        prompt = '''你是一名法律专家。请根据“法律问题”和“标准答案”，判断“待评测答案”是否完全、准确地涵盖了“标准答案”的所有核心要点，由此给出0-10分的评分。你**不用考虑待评测答案表达是否简洁、重点是否突出、是否使用寒暄语、结构是否冗长等非实质性因素**，无须因不够简洁而扣分。

        标准答案：
        {gt_answer}

        待测评答案：
        {model_answer}
        
        以如下格式输出你的结果（中文括号分割，不要换行，不要带括号）：
        评分：；原因：
        '''
        gt_judgments = gt_case['court_information']['ground_truth']['court_judgment']
        count = 0
        judgment_scores = 0
        gt_ans_judgment = ''
        for gtj in gt_judgments:
            if '如不服本判决' not in gtj:
                count = 1
                gt_ans_judgment += f'{count}. {gtj}\n'
                count += 1
        if gt_ans_judgment.endswith('\n'):
            gt_ans_judgment = gt_ans_judgment[:-1]
                
        full_prompt = prompt.format(gt_answer = gt_ans_judgment, model_answer = m_judgement)
        gt_result = func.get_completion(full_prompt, [], flag=0)[0].split('；')[0].replace('评分：','').replace('分','')
        judgment_scores += float(gt_result)/10

        
        # 说理过程
        prompt_reasoning = '''你是一名法律专家。请根据“法律问题”和“标准答案”，判断“待评测答案”是否完全、准确地涵盖了“标准答案”的所有核心要点，由此给出0-10分的评分。你**不用考虑待评测答案表达是否简洁、重点是否突出、是否使用寒暄语、结构是否冗长等非实质性因素**，无须因不够简洁而扣分。

        标准答案：
        {gt_answer}

        待测评答案：
        {model_answer}
        
        以如下格式输出你的结果（中文括号分割，不要换行，不要带括号）：
        评分：；原因：
        '''
        gt_reasoning = gt_case['court_information']['ground_truth']['court_reason'].replace(' ', '').replace('-', '').replace('本院认为：', '')
        full_prompt_reasoning = prompt_reasoning.format(gt_answer = gt_reasoning, model_answer=reasons)
        reason_score = float(func.get_completion(full_prompt_reasoning, [], flag=0)[0].split('；')[0].replace('评分：','').replace('分','').replace('（','').replace('）',''))/10
        
        def remove_article_items(text):
            # 匹配“第x款”，x 可以是中文数字或阿拉伯数字
            pattern = r"第[一二三四五六七八九十百千万\d]+款"
            return re.sub(pattern, '', text)
        
        # 法律依据
        def group_by_law(articles):
            law_dict = defaultdict(list)
            all_law_names = set()

            # 第一次遍历：提取所有法律名称
            for item in articles:
                match = re.match(r"(《[^》]+》)", item)
                if match:
                    law_name = match.group(1)
                    all_law_names.add(law_name)

            # 第二次遍历：分类条文到对应法律
            for item in articles:
                match = re.match(r"(《[^》]+》)", item)
                # 删除第几款的表述
                item = remove_article_items(item)
                if match:
                    law_name = match.group(1)
                    if item != law_name:
                        law_dict[law_name].append(item.replace(law_name, '').replace('《','').replace('》',''))

            # 填补没有条文的法律
            for law_name in all_law_names:
                if law_name not in law_dict:
                    law_dict[law_name] = []

            return dict(law_dict)
        
        def chinese_to_arabic(chinese_num):
            digit_map = {
                '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
                '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
            }
            unit_map = {
                '十': 10, '百': 100, '千': 1000,
                '万': 10000, '亿': 100000000
            }

            result = 0
            section = 0  # 每个万、亿的分节
            number = 0   # 当前的数字
            unit = 1     # 当前的单位

            i = 0
            while i < len(chinese_num):
                char = chinese_num[i]
                if char in digit_map:
                    number = digit_map[char]
                    i += 1
                elif char in unit_map:
                    unit_value = unit_map[char]
                    if unit_value >= 10000:
                        section = (section + number) * unit_value
                        result += section
                        section = 0
                    else:
                        section += number * unit_value
                    number = 0
                    i += 1
                else:
                    # 忽略非法字符
                    i += 1

            return result + section + number
                
        gt_laws = []
        for gtlaw in gt_case['court_information']['ground_truth']['court_law']:
            if '、' in gtlaw:
                lname = re.match(r"(《[^》]+》)", gtlaw).group(1)
                tt = gtlaw.split('、')
                for ll in tt:
                    if lname not in ll:
                        gt_laws.append(lname + ll)
                    else:
                        gt_laws.append(ll)
            else:
                gt_laws.append(gtlaw)
            
        laws = group_by_law(gt_laws)
        total_score = 0.0

        # 删除第几款的表述
        m_laws = remove_article_items(model_judgment)
            
        for law_name, articles in laws.items():
            if law_name in m_laws:
                for article in articles:
                    rest2 = '第' + str(chinese_to_arabic(article.replace('第','').replace('条',''))) + '条'
                    if article in m_laws or rest2 in m_laws:
                        total_score += 1
                
                    
        max_score = 0.0
        for articles in laws.values():
            if articles:
                max_score += len(articles)
            else:
                max_score += 1
        # 衡量法条数够不够
        law_ratio = total_score / max_score if max_score > 0 else 0.0


                
        JUD_REA_LAW = {
            'REA': {'model_answer': reasons, 'ground_truth': gt_reasoning, 'score': reason_score },
            'LAW': {'model_answer': m_laws, 'ground_truth': gt_laws, 'score': law_ratio},
            'JUD': {'model_answer': m_judgement, 'ground_truth': gt_ans_judgment,'score': judgment_scores},
        }
        return JUD_REA_LAW
    
    def evaluate(self, model, case, model_result):
        dialog_history = model_result['dialog_history']
        judge_dialog = [dh for dh in dialog_history if dh['role'] == 'Judge']
        judge_keywords = ['本院认为', '最终宣判', '最终判决']
        flag = False
        id = case['id']
        # 将最靠近对话末尾且同时出现“本院认为”和“判决”关键词的法官发言作为判决文本用于评分
        for dialog in reversed(judge_dialog):
            if dialog['role'] == 'Judge' and ('本院认为' in dialog['content'] and '判决' in dialog['content']):
                model_judgment = dialog['content']
                flag = True
                break

        def count_d(text):
            c = 0
            for t in text:
                if t == '第':
                    c += 1
            return c > 20
        
            
        if flag and ('公诉人' in model_judgment or '模拟被告回应' in model_judgment):
            flag = False
        
        if flag and count_d(model_judgment):
            flag = False
        
        if not flag:
            result = {
                'REA': 0,
                'LAW': 0,
                'JUD': 0,
                'PFS': {
                    'UNI': 0,
                    'STA': 0,
                    'ACT': 0,
                    'PFS': 0
                },
                'dialog_history': dialog_history
            }
            func.save_json(result, os.path.join(self.intermediate_eval,  model, 'CI',  f'{id}.json'))
            return None
        PFS = self.get_PFS(dialog_history)
        if PFS['UNI'] == 1:
            JUD_REA_LAW = self.get_JUD_REA_LAW(model_judgment, case)
        else:
            JUD_REA_LAW = {
                'REA': 0,
                'LAW': 0,
                'JUD': 0
            }
        
        result = {
            'REA': JUD_REA_LAW['REA'],
            'LAW': JUD_REA_LAW['LAW'],
            'JUD': JUD_REA_LAW['JUD'],
            'PFS': PFS,
            'dialog_history': dialog_history
        }
        
        func.save_json(result, os.path.join(self.intermediate_eval,  model, 'CI',  f'{id}.json'))
    
    def get_final_score(self, intermediate_model_folder, model):
        REA = []
        LAW = []
        JUD = []
        PFS = []
        UNI = []
        STA = []
        ACT = []
        for file_name in os.listdir(intermediate_model_folder):
            data = func.load_json(os.path.join(intermediate_model_folder, file_name))
            if data['REA']!=0:
                rea_score = data['REA']['score']
            else:
                rea_score = 0
            if data['LAW']!=0:
                law_score = data['LAW']['score']
            else:
                law_score = 0
            if data['JUD'] != 0:
                jud_score = data['JUD']['score']
            else:
                jud_score = 0
            uni_score = data['PFS']['UNI']
            sta_score = data['PFS']['STA']
            act_score = data['PFS']['ACT']
                
            pfs_score = data['PFS']['UNI'] * np.mean([data['PFS']['STA'], data['PFS']['ACT']])
            
            turn = data['dialog_history'][-1]['turn']
            if turn < 4:
                rea_score = 0
                law_score = 0
                jud_score = 0
                uni_score = 0
                sta_score = 0
                act_score = 0
                pfs_score = 0
            
            REA.append(rea_score)
            LAW.append(law_score)
            JUD.append(jud_score)
            PFS.append(pfs_score)
            UNI.append(uni_score)
            STA.append(sta_score)
            ACT.append(act_score)
        print('REA', np.mean(REA))
        print('LAW', np.mean(LAW))
        print('JUD', np.mean(JUD))
        print('PFS', np.mean(PFS))
        print('UNI', np.mean(UNI))
        print('STA', np.mean(STA))
        print('ACT', np.mean(ACT))
        result = {
            'model': model,
            'REA': np.mean(REA),
            'LAW': np.mean(LAW),
            'JUD': np.mean(JUD),
            'STA': np.mean(STA),
            'ACT': np.mean(ACT),
            'UNI': np.mean(UNI),
            'PFS': np.mean(PFS)
        }
        func.save_json(result, os.path.join(self.final_eval, f'{model}_final.json'))
    
    def iterate(self):
        for model in os.listdir(self.dialog_history_dir):
            print(f'Processing {model}.')
            intermediate_model_folder = os.path.join(self.intermediate_eval,  model, 'CI')
            if not os.path.exists(intermediate_model_folder):
                os.makedirs(intermediate_model_folder)
                
            processed_ids = []
            for file_name in os.listdir(intermediate_model_folder):
                if file_name.endswith('.json'):
                    id = file_name.split('-')[-1].replace('.json','')
                    processed_ids.append(int(id))


            self.dialog_history = func.load_jsonl(os.path.join(self.dialog_history_dir, model,  'CI_dialog_history.jsonl'))
            for dh in self.dialog_history:
                case_id = int(dh['case_id'].split('-')[-1])
                if case_id in processed_ids:
                    print(f'Case {case_id} already evaluated.')
                    continue
                print(f'Evaluating case {case_id}.')
                for case in self.ground_truth:
                    if int(case['id'].split('-')[-1]) == case_id:
                        break
                self.evaluate(model, case, dh)
            self.get_final_score(intermediate_model_folder, model)

def main():
    parser = argparse.ArgumentParser()
    CIEvaluator.add_parser_args(parser)
    args = parser.parse_args()
    evaluator = CIEvaluator(args)
    evaluator.iterate()
    
if __name__ == '__main__':
    main()