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


class CREvaluator:
    def __init__(self, args):
        self.ground_truth = func.load_jsonl('./src/data/case/J1-Eval_CR.jsonl')
        self.args = args
        
        self.dialog_history_dir = args.dialog_history_dir
        self.intermediate_eval = args.intermediate_eval
        self.final_eval = args.final_eval
        
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument("--dialog_history_dir", default = "./src/data/dialog_history", type = str)
        parser.add_argument("--intermediate_eval", default = "./src/Eval/eval_result", type = str)
        parser.add_argument("--final_eval", default = "./src/Eval/final_result/CR", type = str)    
        
    def get_PFS(self, dialog_history):
        procedures = {
            '开庭审理': {
                '前科劣迹': {
                    'options': ['前科劣迹'],
                    'flag': False
                },
                '审判程序': {
                    'options': ['适用程序', '程序'],
                    'flag': False
                },
                '明确罪名': {
                    'options': ['罪名是', '明确罪名'],
                    'flag': False
                },
                '申请回避': {
                    'options': ['申请回避'],
                    'flag': False
                    },
                # '诉讼权利': {
                #     'options': ['诉讼权利','告知双方当事人在诉讼中的权利'],
                #     'flag' : False
                # },
                '核实身份': {
                    'options': ['核实'],
                    'flag' : False
                },
                '认罪认罚': {
                    'options': ['认罪认罚'],
                    'flag' : False
                },
            },
            '法庭调查': {
                    '法庭询问': {
                        'options': ['询问被告', '法庭询问', '进行讯问'],
                        'flag': False
                    }
                }
            }
        
        UNI = 0
        STA = 0
        ACT = 0
        judge_dialog = [dh for dh in dialog_history if dh['role'] == 'Judge']
        for jd in dialog_history:
            content = jd['content']
            role = jd['role']
            if role == 'Judge':
                # 对话成立性
                count = 0
                if '<开庭审理>' in content and '<结束庭审>' in content:
                    PFS = {
                        'UNI': 0,
                        'STA': 0,
                        'ACT': 0
                    }
                    return PFS
            
                
            
        # 关键步骤正确性
        scored_procedures = set()
        for idx, jd in enumerate(dialog_history):
            content = jd['content']
            role = jd['role']
            if role == 'Judge':
                UNI = 1
                if idx < len(dialog_history) - 1:
                    next_turn = dialog_history[dialog_history.index(jd) + 1]
                    if next_turn['role'] == 'Defendant' or next_turn['role'] == 'Lawyer' or next_turn['role'] == 'Procurator':
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
            'UNI': UNI,
            'STA': STA/2, # 每个阶段的关键词是否完整
            'ACT': ACT/total_ACT, # 所有阶段的关键词
            'check': procedures
        }
        return PFS
    
    def get_CRI_SEN_FINE_REA_LAW(self, model_judgment, gt_case):
        # 模型判决
        judgment_patterns = [
            r'判决如下：(.*?)依照法条',
            r'判决如下[:：](.*?)(?=\n\n<结束庭审>)',  # 匹配“判决如下”后的内容，直到 <结束庭审>
            r'判处(.*?)(?=\n\n<结束庭审>)',  # 匹配“判处”后的内容，直到 <结束庭审>
            r'判决如下：(.*?)(?=\n\n)',  # 另一种方式，不强求换行符
            r'判决如下：(.*?)依照',
            r'判决如下：(.*?)\n依照法条'
        ]

        m_judgement = None
        for pattern in judgment_patterns:
            match = re.search(pattern, model_judgment, re.DOTALL)
            if match:
                m_judgement = match.group(1).strip()
                break
        if m_judgement is None or len(m_judgement)==0:
            m_judgement = model_judgment
            
        judgement = m_judgement.replace(' ','').replace('-','').split('\n')
        
        # 模型说理
        reason_patterns = [
            r'本院认为：(.*?)依照',
            r'本院认为，(.*?)判决如下：',
            r'本院认为[：，]?(.*?)(?=[依照，根据])',
            r'本院认为[：，]?(.*?)(?=根据)',
            r'本院认为：\n\n(.*?)\n\n根据',
            r'本院认为，(.*?)\n\n依照',
            r'本院认为，(.*?)\n\n根据',
            r'本庭认为，(.*?)判决如下：'
        ]
        reasons = None
        for pattern in reason_patterns:
            reasons = re.search(pattern, model_judgment.replace('本庭认为','本院认为'), re.DOTALL)
            if reasons:
                reasons = reasons.group(1).strip()
                break
            else:
                continue
        if reasons is None or len(reasons)==0:
            reasons = model_judgment
        
        # 判断罪名
        crime = re.findall(r'犯(.*?)罪', gt_case['court_information']['ground_truth']['court_judgment'][0])[0]
        if crime in reasons + '\n'.join(judgement):
            crime_score = 1
        else:
            crime_score = 0
        
        # 中文数字映射表
        CN_NUM = {
            '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
        }
        CN_UNIT = {'十': 10, '百': 100, '千': 1000}

        def chinese_to_num(cn: str) -> int:
            if not cn:
                return 0
            if cn.isdigit():
                return int(cn)
            
            num = 0
            unit = 1
            result = 0
            for char in reversed(cn):
                if char in CN_UNIT:
                    unit = CN_UNIT[char]
                    if num == 0:
                        num = 1 
                elif char in CN_NUM:
                    result += CN_NUM[char] * unit
                    unit = 1
                else:
                    continue
            return result + num * unit 

        def extract_sentence_length(text: str) -> int:
            if '有期徒刑' in text and '年' in text and '个月' in text:
                pattern = r"有期徒刑(?:(?P<year>[一二三四五六七八九十零两百千\d]+)年)?(?:零|又)?(?P<month>[一二三四五六七八九十零\d]*)个月?"
            elif '有期徒刑' in text and '年' in text and '个月' not in text:
                pattern = r"有期徒刑(?:(?P<year>[一二三四五六七八九十零两百千\d]+)年)"
            elif '有期徒刑' in text and '年' not in text and '个月' in text:
                pattern = r"有期徒刑(?:(?P<month>[一二三四五六七八九十零两百千\d]+)个月)"
            elif '拘役' in text and '年' in text and '个月' in text:
                pattern = r"拘役(?:(?P<year>[一二三四五六七八九十零两百千\d]+)年)?(?:零|又)?(?P<month>[一二三四五六七八九十零\d]*)个月?"
            elif '拘役' in text and '年' in text and '个月' not in text:
                pattern = r"拘役(?:(?P<year>[一二三四五六七八九十零两百千\d]+)年)"
            elif '拘役' in text and '年' not in text and '个月' in text:
                pattern = r"拘役(?:(?P<month>[一二三四五六七八九十零两百千\d]+)个月)"
            else:
                return 0
            match = re.search(pattern, text)
            if match:
                groups = match.groupdict()
                if 'year' in groups and groups['year'] is not None:
                    year_str = match.group("year") or ""
                else:
                    year_str = ''
                if 'month' in groups and groups['month'] is not None:
                    month_str = match.group("month") or ""
                else:
                    month_str = ''
                
                # 处理年和月的转换
                years = chinese_to_num(year_str) if year_str else 0
                months = chinese_to_num(month_str) if month_str else 0
                
                return years * 12 + months  # 年转为月并加上月
            return 0


        
        model_sentence = extract_sentence_length(model_judgment)
        flag = False
        for gts in gt_case['court_information']['ground_truth']['court_judgment']:
            gt_sentence = extract_sentence_length(gts)
            if gt_sentence:
                flag = True
                break
        
        if flag:
            # 修改
            # sentence_score = np.abs(np.log(1+model_sentence) - np.log(1+gt_sentence))
            sentence_diff = np.abs(np.log(1+model_sentence) - np.log(1+gt_sentence))
            sentence_score = 1 / (1 + sentence_diff)
        else:
            sentence_score = '无'
        
        def chinese_money_to_num(chinese: str) -> int:
            cn_num = {
                '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
                '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
            }
            cn_unit = {
                '十': 10, '百': 100, '千': 1000,
                '万': 10000, '亿': 100000000
            }

            result = 0       # 最终结果
            section = 0      # 每个“万”段的结果
            number = 0       # 当前数字
            unit = 1         # 当前单位

            for char in chinese:
                if char in cn_num:
                    number = cn_num[char]
                elif char in cn_unit:
                    unit = cn_unit[char]
                    if unit >= 10000:
                        section = (section + number) * unit
                        result += section
                        section = 0
                    else:
                        section += (number if number != 0 else 1) * unit
                    number = 0
                else:
                    continue

            total = result + section + number
            return total

        def extract_fine_amount(text: str) -> int:
            pattern =  r"(?:处罚金|罚金)[：:，]?(?:人民币)?\s*([一二三四五六七八九十百千万零两\d]+(?:[\.．点][\d零一二三四五六七八九十]+)?)(?:元|块)?"
            #pattern = r"(?:处罚金|罚金)[\s\S]{0,10}?人民币[\s\S]{0,10}?([一二三四五六七八九十百千万零两\d]+)[\s\S]{0,5}?元[。．.]?"
            text = re.sub(r'\s+', '', text)
            match = re.search(pattern, text.replace(' ',''), flags=re.DOTALL)
            if match:
                amount_text = match.group(1)
                print(f"匹配到金额文本：{amount_text}")
                if amount_text.isdigit():
                    return int(amount_text)
                else:
                    return chinese_money_to_num(amount_text)
            else:
                print("⚠️ 没有匹配到罚金字段")
            return 0

        def remove_commas_from_numbers(text: str) -> str:
            pattern = r'\d{1,3}(?:[，,]\d{3})*'
            return re.sub(pattern, lambda x: x.group(0).replace(',', '').replace('，', ''), text)
        model_judgment = remove_commas_from_numbers(model_judgment)
        model_fine = extract_fine_amount(model_judgment)
        gt_sentence = gt_case['court_information']['ground_truth']['court_judgment']
        flag = False
        for gts in gt_sentence:
            gt_fine = extract_fine_amount(gts.replace(',',''))
            if gt_fine:
                flag = True
                break
        # 修改
        # fine_score = np.abs(np.log(1+model_fine) - np.log(1+gt_fine))
        fine_diff = np.abs(np.log(1+model_fine) - np.log(1+gt_fine))
        fine_score = 1 / (1 + fine_diff)
        
        # 整体判决
        gt_sentences = ''
        count = 1
        for gts in gt_sentence:
            gt_sentences += f'{count}. {gts}'
            count += 1
        
        
        # 说理过程
        prompt_reasoning = '''你是一名法律专家。请根据“法律问题”和“标准答案”，判断“待评测答案”是否完全、准确地涵盖了“标准答案”的所有核心要点，由此给出0-10分的评分。你**不用考虑待评测答案表达是否简洁、重点是否突出、是否使用寒暄语、结构是否冗长等非实质性因素**，无须因不够简洁而扣分。

        标准答案：
        {gt_answer}

        待测评答案：
        {model_answer}
        
        以如下格式输出你的结果（中文括号分割，不要换行，不要带括号）：
        评分：；原因：
        '''
        gt_reasoning = gt_case['court_information']['ground_truth']['court_reason']
        full_prompt_reasoning = prompt_reasoning.format(gt_answer = gt_reasoning, model_answer=reasons)
        reason_score = float(func.get_completion(full_prompt_reasoning, [], flag=0)[0].split('；')[0].replace('评分：','').replace('分',''))/10
        
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
                item = remove_article_items(item)
                if match:
                    law_name = match.group(1)
                    if item != law_name:
                        law_dict[law_name].append(item.replace(law_name, '').replace('《','').replace('》','').replace('及','').replace('、',''))

            # 填补没有条文的法律
            for law_name in all_law_names:
                if law_name not in law_dict:
                    law_dict[law_name] = []

            return dict(law_dict)
        
        
    
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
                
        ratio = total_score / max_score if max_score > 0 else 0.0
        
        CRI_SEN_FINE_REA_LAW = {
            'CRI': {'model_answer': reasons + '\n'.join(judgement), 'ground_truth': crime, 'score': crime_score},
            'VER': {
                'SEN': {'model_answer':model_sentence, 'ground_truth':gt_sentence, 'score':sentence_score},
                'FINE': {'model_answer':model_fine,'ground_truth':gt_fine,'score':fine_score},
            },
            'REA': {'model_answer': reasons, 'ground_truth': gt_reasoning, 'score': reason_score},
            'LAW': {'model_answer': m_laws, 'ground_truth': laws, 'score': ratio}
        }
        return CRI_SEN_FINE_REA_LAW
    
    def evaluate(self, model, case, model_result):
        dialog_history = model_result['dialog_history']
        judge_dialog = [dh for dh in dialog_history if dh['role'] == 'Judge']
        judge_keywords = ['结束庭审', '最终宣判', '最终判决']
        jud_flag = False
        for dialog in reversed(judge_dialog):
            if dialog['role'] == 'Judge' and ('本院认为' in dialog['content'] and '判决' in dialog['content']):
                model_judgment = dialog['content']
                jud_flag = True
                break
        
        for dialog in reversed(judge_dialog):
            if dialog['role'] == 'Judge' and dialog['content'] == '':
                jud_flag = False
        
        if jud_flag and '本院认为：；判决如下：；依照法条：；结束庭审' in model_judgment:
            jud_flag = False
        
        if jud_flag:
            PFS = self.get_PFS(dialog_history)
            if PFS['UNI'] == 1:
                CRI_SEN_FINE_REA_LAW = self.get_CRI_SEN_FINE_REA_LAW(model_judgment, case)
            else: 
                CRI_SEN_FINE_REA_LAW = {
                    'CRI': 0,
                    'VER': {
                    'SEN': 0,
                    'FINE': 0
                    },
                    'REA': 0,
                    'LAW': 0
                }
            result = {
                'PFS': PFS,
                'CRI': CRI_SEN_FINE_REA_LAW['CRI'],
                'VER': {
                    'SEN': CRI_SEN_FINE_REA_LAW['VER']['SEN'],
                    'FINE': CRI_SEN_FINE_REA_LAW['VER']['FINE']
                },
                'REA': CRI_SEN_FINE_REA_LAW['REA'],
                'LAW': CRI_SEN_FINE_REA_LAW['LAW'],
                'dialog_history': dialog_history
            }
        else:
            result = {
                #修改
                'PFS': {"UNI":0,"STA":0,"ACT":0},
                'CRI': 0,
                'VER': {
                    'SEN': 0,
                    'FINE': 0
                },
                'REA': 0,
                'LAW': 0,
                'dialog_history': dialog_history
            }
        id = case['id']
        func.save_json(result, os.path.join(self.intermediate_eval,  model, 'CR',  f'{id}.json'))
    
    
    def get_final_score(self, intermediate_model_folder, model):
        CRI = []
        # SEN_1 = []
        # SEN_2 = []
        # FINE_1 = []
        # FINE_2 = []
        SEN = []
        FINE = []
        REA = []
        LAW = []
        PFS = []
        STA = []
        UNI = []
        ACT = []
        for file_name in os.listdir(intermediate_model_folder):
            data = func.load_json(os.path.join(intermediate_model_folder, file_name))

            dialog = data['dialog_history']
            turn = dialog[-1]['turn']
            
            if data['CRI'] != 0:
                cri = data['CRI']['score']
            else:
                cri = 0
            # if data['VER']['SEN'] != 1 and turn > 4:
            #     sen = data['VER']['SEN']['score']
            #     SEN_1.append(sen)
            # else:
            #     sen = 0
            #     SEN_2.append(sen)
            # if data['VER']['FINE'] != 1 and turn > 4:
            #     fine = data['VER']['FINE']['score']
            #     FINE_1.append(fine)
            # else:
            #     FINE_2.append(1)
            # 处理 SEN 分值，既兼容 dict 又兼容 int
            if isinstance(data['VER']['SEN'], dict):
                sen = data['VER']['SEN'].get('score', 0)
            else:
                sen = data['VER']['SEN']  # 可能是 0 或 1
            SEN.append(sen)
            # 处理 FINE 分值
            if isinstance(data['VER']['FINE'], dict):
                fine = data['VER']['FINE'].get('score', 0)
            else:
                fine = data['VER']['FINE']
            FINE.append(fine)

            if data['REA'] != 0:
                rea = data['REA']['score']
            else:
                rea = 0
            if data['LAW'] != 0:
                law = data['LAW']['score']
            else:
                law = 0
            sta = data['PFS']['STA']
            act = data['PFS']['ACT']
            uni = data['PFS']['UNI']
            pfs = data['PFS']['UNI'] * np.mean([data['PFS']['STA'], data['PFS']['ACT']])
            
            if turn < 4:
                rea = 0
                law = 0
                uni = 0
                sta = 0
                act = 0
                pfs = 0
                cri = 0
            
            CRI.append(cri)
            REA.append(rea)
            LAW.append(law)
            PFS.append(pfs)
            STA.append(sta)
            ACT.append(act)
            UNI.append(uni)
        
        def min_max_normalize(values):
            import numpy as np

            values = np.array(values, dtype=np.float64)
            v_min, v_max = values.min(), values.max()

            if v_max == v_min:
                return [0.0 for _ in values] 
            else:
                normalized = (values - v_min) / (v_max - v_min)
                return normalized.tolist()

        # if len(SEN_1) > 0:
        #     SEN_1 = min_max_normalize(SEN_1)
        # SEN = SEN_1 + SEN_2
        # if len(FINE_1) > 0:
        #     FINE_1 = min_max_normalize(FINE_1)
        # FINE = FINE_1 + FINE_2
        # SEN = SEN_1 + SEN_2
        # FINE = FINE_1 + FINE_2
        print('CRI', np.mean(CRI))
        print('SEN', np.mean(SEN))
        print('FINE', np.mean(FINE))
        print('REA', np.mean(REA))
        print('LAW', np.mean(LAW))
        print('PFS', np.mean(PFS))
        print('STA', np.mean(STA))
        print('ACT', np.mean(ACT))
        print('UNI', np.mean(UNI))
        result = {
            'model': model,
            'PFS': {
                'STA': np.mean(STA),
                'ACT': np.mean(ACT),
                'UNI' :np.mean(UNI),
                'PFS' :np.mean(PFS)
            },
            'CRI': np.mean(CRI),
            'VER': {
                'SEN': np.mean(SEN),
                'FINE': np.mean(FINE),
                'VER': np.mean([np.mean(SEN), np.mean(FINE)])
            },
            'REA': np.mean(REA),
            'LAW': np.mean(LAW)
        }
        func.save_json(result, os.path.join(self.final_eval, f'{model}_final.json'))
    
    
    
    def iterate(self):
        for model in os.listdir(self.dialog_history_dir):
            print(f'Processing {model}.')
            intermediate_model_folder = os.path.join(self.intermediate_eval,  model, 'CR')
            if not os.path.exists(intermediate_model_folder):
                os.makedirs(intermediate_model_folder)
                
            processed_ids = []
            for file_name in os.listdir(intermediate_model_folder):
                if file_name.endswith('.json'):
                    case_id = file_name.split('-')[-1].replace('.json','')
                    processed_ids.append(int(case_id))
            

            self.dialog_history = func.load_jsonl(os.path.join(self.dialog_history_dir, model,  'CR_dialog_history.jsonl'))
            
            for dh in self.dialog_history:
                case_id = int(dh['case_id'].split('-')[-1])
                if case_id in processed_ids:
                    print(f'Case {case_id} has been processed.')
                    continue
                print(f'Evaluating case {case_id}.')
                for case in self.ground_truth:
                    if int(case['id'].split('-')[-1]) == case_id:
                        break
            
                self.evaluate(model, case, dh)
            self.get_final_score(intermediate_model_folder, model)

def main():
    parser = argparse.ArgumentParser()
    CREvaluator.add_parser_args(parser)
    args = parser.parse_args()
    evaluator = CREvaluator(args)
    evaluator.iterate()
    
if __name__ == '__main__':
    main()