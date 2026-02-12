import re
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(parent_dir)
import utils.utils_func as func
import argparse

class DDEvaluator:
    def __init__(self, args):
        self.ground_truth = func.load_jsonl('./src/data/case/J1-Eval_DD.jsonl')
        self.args = args
        
        self.dialog_history_dir = args.dialog_history_dir
        self.intermediate_eval = args.intermediate_eval
        self.final_eval = args.final_eval
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument("--dialog_history_dir", default = "./src/data/dialog_history", type = str)
        parser.add_argument("--intermediate_eval", default = "./src/Eval/eval_result", type = str)
        parser.add_argument("--final_eval", default = "./src/Eval/final_result/DD", type = str)
    
    def _clean(self, text):
        clean_words = ['。', '法定代表人：', '住所：', '住', ' ']
        for word in clean_words:
            text = text.replace(word, '')
        return text
    
    def reformat(self, text, defendant_type):
        defendant_patterns = [
            r"答辩人（?(?:如果是)?(?:自然人|法人|自然人或法人)?）?:?(.*?)\n"
        ]
        defendant_info = None
        for pattern in defendant_patterns:
            match = re.search(pattern, text.replace('*',''), re.DOTALL)
            if match:
                defendant_info = match.group(1).strip()
                break
        if defendant_info is None or len(defendant_info) == 0:
            defendant = text
        else:
            defendant = defendant_info
        
        # 提取答辩意见
        defense_patterns = [
            r"(?:答辩意见|答辩如下|如下答辩|我方提出如下答辩)：\n*?(.*?)(?=\n*(?:此致|证据和证据来源|四、证据及证据|证据|\n*答辩人：|<询问结束>|$))"
        ]
        defense_info = None
        for defense_pattern in defense_patterns:
            defense_info = re.search(defense_pattern, text.replace(' ','').replace('*',''), re.DOTALL)
        if defense_info is None or len(defense_info[0]) == 0:
            defense = text
        else:
            defense = defense_info.group(1).replace(' ','')

        # 提取证据
        evidence_patterns = [
            r"证据和证据来源(?:，证人姓名和住所)?:\n?(.*?)(?=\n{2,}此致|\n{2,}|$)"
        ]
        evidence_info = None
        for evidence_pattern in evidence_patterns:
            evidence_info = re.search(evidence_pattern, text.replace(' ',''), re.DOTALL)
        
        if evidence_info is None or len(evidence_info) == 0:
            evidence = text
        else:
            evidence = evidence_info.group(1).replace(' ','')
                
        model_answer = {
            'defendant': defendant
        }
                
        model_answer['defense'] = defense
        model_answer['evidence'] = evidence
        return model_answer
    
    def compare_defendant(self, defendant_type, ground_truth, model_answer, original_model_answer):
        defendant = ground_truth['specific_characters']['defendant']
        defense = ground_truth['statement_of_defence']
        evidence = ground_truth['evidence']
        
        
        # 统计被告
        model_defence = model_answer['defendant']
        defen = 0
        t_score = 0
        if defendant_type == 'personal':
            gt_name = defendant['name']
            gt_sex = defendant['gender']
            gt_birth = defendant['birth_date']
            gt_address = defendant['address']
            gt_ethnicity = defendant['ethnicity']
            if gt_name:
                if gt_name in model_defence:
                    defen += 1
                t_score += 1
            if gt_sex:
                if gt_sex in model_defence and '男/女' not in model_defence:
                    defen += 1
                t_score += 1
            if gt_birth:
                if gt_birth in model_defence:
                    defen += 1
                t_score += 1
            if gt_address:
                if gt_address in model_defence:
                    defen += 1
                t_score += 1
            if gt_ethnicity:
                if gt_ethnicity in model_defence:
                    defen += 1
                t_score += 1
        else:
            gt_name = defendant['name']
            gt_address = defendant['address']
            gt_representative = defendant['representative']
            if gt_name:
                if gt_name in model_defence:
                    defen += 1
                t_score += 1
            if gt_address:
                if gt_address in model_defence:
                    defen += 1
                t_score += 1
            if gt_representative:
                if gt_representative in model_defence:
                    defen += 1
                t_score += 1
        RES_score = defen / t_score
        
        # 统计答辩意见
        prompt_defense = '''你是一名法律专家。请根据“法律问题”和“标准答案”，判断“待评测答案”是否完全、准确地涵盖了“标准答案”的所有核心要点，由此给出0-10分的评分。你**不用考虑待评测答案表达是否简洁、重点是否突出、是否使用寒暄语、结构是否冗长等非实质性因素**，无须因不够简洁而扣分。

        标准答案：
        {gt_answer}

        待测评答案：
        {model_answer}
        
        以如下格式输出你的结果（中文括号分割，不要换行，不要带括号）：
        评分：；原因：
        '''
        
        model_defence = model_answer['defense'].replace(' ','')
        full_prompt_defense = prompt_defense.format(gt_answer = defense, model_answer = model_defence)
        try:
            DEF_score = float(func.get_completion(full_prompt_defense, [], flag=0)[0].split('；')[0].replace('评分：','').replace('分',''))/10
        except:
            full_prompt_defense = prompt_defense.format(gt_answer = defense, model_answer = original_model_answer)
            DEF_score = float(func.get_completion(full_prompt_defense, [], flag=0)[0].split('；')[0].replace('评分：','').replace('分',''))/10
        
        # 统计证据
        prompt_evi = '''你是一名法律专家。请根据“法律问题”和“标准答案”，判断“待评测答案”是否完全、准确地涵盖了“标准答案”的所有核心要点，由此给出0-10分的评分。你**不用考虑待评测答案表达是否简洁、重点是否突出、是否使用寒暄语、结构是否冗长等非实质性因素**，无须因不够简洁而扣分。

        标准答案：
        {gt_answer}

        待测评答案：
        {model_answer}
        
        以如下格式输出你的结果（中文括号分割，不要换行，不要带括号）：
        评分：；原因：
        '''
        
        model_evidence = model_answer['evidence']
        gt_evidences = []
        for de in evidence.keys():
            gt_evidences.append(evidence[de]['evidence'].replace('\n','、'))
        
        evi_scores = 0
        total_evi_scores = 0
        
        if len(gt_evidences) > 0:
            gtevidence = ''
            count = 1
            for gte in gt_evidences:
                gtevidence += f'{count}. {gte}\n'
                count += 1
                
            full_prompt_evi = prompt_evi.format(gt_answer=gtevidence, model_answer=model_evidence)
            temp = float(func.get_completion(full_prompt_evi, [], flag=0)[0].split('；')[0].replace('评分：','').replace('分',''))/10
            evi_scores += temp
            total_evi_scores += 1
        else:
            gtevidence = '无相关证据'
            full_prompt_evi = prompt_evi.format(gt_answer=gtevidence, model_answer=model_evidence)
            temp = float(func.get_completion(full_prompt_evi, [], flag=0)[0].split('；')[0].replace('评分：','').replace('分',''))/10
            evi_scores += temp
            total_evi_scores += 1
        evi_score = evi_scores/total_evi_scores
       
        DOC = {
            "RES": {'RES': RES_score, 'model_RES': model_answer['defendant'], 'ground_truth': defendant},
            "DEF": {'DEF': DEF_score, 'model_DEF': model_defence, 'ground_truth': defense },
            "EVI": {'EVI': evi_score, 'model_EVI': model_evidence, 'ground_truth': gt_evidences, },
            "AVE": np.mean([RES_score, DEF_score, evi_score])
        }
        return DOC
    
    def format_item_names(self, name_set, correct_name, model_defence):
        for item in name_set:
            model_defence = model_defence.replace(item, correct_name)
        return model_defence
    
    def format_following_score(self, model_defence):
        # 名称正确性得分
        label_score = 0
        for i in ['答辩人：', '证据和证据来源，证人姓名和住所：']:
            if i in model_defence:
                label_score += 1
        defense_pattern = r'对(.*?)人民法院（(.*?)）(.*?)民初(.*?)号(.*?)一案的起诉，答辩如下：'
        defense = re.findall(defense_pattern, model_defence)
        if defense != []:
            label_score += 1
        
        # 顺序得分
        incorrect_defendant_names = [
            '答辩人（自然人）：',
            '答辩人（自然人或法人）：',
            '答辩人（如果是自然人）：',
            '答辩人（如果是法人）：',
            '答辩人（法人）：'
            ]
        model_defence = self.format_item_names(incorrect_defendant_names, '答辩人：', model_defence)
        
        # 修改：有逻辑错误
        # last_pos = -1
        # for char in ['答辩人：', '答辩如下：', '证据和证据来源，证人姓名和住所：']:
        #     if char not in model_defence:
        #         sequential_score = 0
        #     current_pos = model_defence.find(char, last_pos + 1)
        #     if current_pos == -1:
        #         sequential_score = 0
        #     last_pos = current_pos
        # sequential_score = 1

        # 二次修改，reform_model_defence之后再判断顺序
        # keys = ['答辩人：', '答辩如下：', '证据和证据来源，证人姓名和住所：']
        # positions = []

        # sequential_score = 1 

        # for key in keys:
        #     pos = model_defence.find(key)
        #     if pos == -1:
        #         sequential_score = 0 
        #         break
        #     positions.append(pos)

        # # 如果三个字段都存在，再检查它们是否按升序排列
        # if sequential_score == 1:
        #     if not (positions[0] < positions[1] < positions[2]):
        #         sequential_score = 0        

        last_pos = -1
        sequential_score = -1
        for char in ['答辩人：', '答辩如下：', '证据和证据来源，证人姓名和住所：']:
            if char not in model_defence:
                sequential_score = 0
            current_pos = model_defence.find(char, last_pos + 1)
            if current_pos == -1:
                sequential_score = 0
            last_pos = current_pos
        if sequential_score == -1:
            sequential_score = 1

        # 修改 label_score总分为3
        total_score = sequential_score * (label_score/3)
        FOR = {
            "label_score": label_score,
            "sequential_score": sequential_score,
            "AVE": total_score,
            'model_defence': model_defence
        }
        
        return FOR
    
    def evaluate(self, model, defendant_dh, defendant_type, ground_truth):
        id = defendant_dh['case_id']
        flag = False
        for dialog in reversed(defendant_dh['dialog_history']):
            if dialog['role'] == 'Lawyer' and dialog['content'] == None:
                flag = False
                break
            # 修改
            if dialog['role'] == 'Lawyer' and ('民事答辩状' in dialog['content'] or '答辩人' in dialog['content']):
                model_defence = dialog['content']
                flag = True
                break
            
            
        if not flag:
            print(f'Case {id} unfinished task!')
            evaluation_result = {
                'DOC': {
                    # 修改
                    "RES": {'RES': 0, 'model_defence': 0, 'ground_truth': 0},
                    "DEF": {'DEF': 0, 'model_defence': 0, 'ground_truth': 0 },
                    "EVI": {'EVI': 0, 'model_evidence': 0, 'ground_truth': 0},
                    "AVE": np.mean([0, 0, 0])
                },
                'FOR': {
                    "label_score": 0,
                    "sequential_score": 0,
                    "FOR": 0,
                    "AVE": 0,
                    'model_defence': dialog['content'] # 修改
                },
                "AVE": np.mean([0, 0]),
                "dialog_history": defendant_dh['dialog_history']
            }
            func.save_json(evaluation_result, os.path.join(self.intermediate_eval,  model, 'DD',  f'{id}.json'))
            return None
        model_answer = self.reformat(model_defence, defendant_type)
        DOC = self.compare_defendant(defendant_type, ground_truth, model_answer, model_defence)
        FOR = self.format_following_score(model_defence)
        evaluation_result = {
            "DOC": DOC,
            "FOR": FOR,
            "AVE": np.mean([DOC['AVE'], FOR['AVE']])
        }
        
        func.save_json(evaluation_result, os.path.join(self.intermediate_eval,  model, 'DD',  f'{id}.json'))
    
    def get_final_score(self, intermediate_model_folder, model):
        RES = []
        DEF = []
        EVI = []
        FOR = []
        for file_name in os.listdir(intermediate_model_folder):
            print(file_name)
            data = func.load_json(os.path.join(intermediate_model_folder, file_name))
            RES_score = data['DOC']['RES']['RES']
            DEF_score = data['DOC']['DEF']['DEF']
            evi_score = data['DOC']['EVI']['EVI']
            for_score = data['FOR']['AVE']
            RES.append(RES_score)
            DEF.append(DEF_score)
            EVI.append(evi_score)
            FOR.append(for_score)
        print('RES', np.mean(RES))
        print('DEF', np.mean(DEF))
        print('EVI', np.mean(EVI))
        print('FOR', np.mean(FOR)) 
        result = {
            'model': model,
            'RES': np.mean(RES),
            'DEF': np.mean(DEF),
            'EVI': np.mean(EVI),
            'DOC': np.mean([np.mean(RES), np.mean(DEF), np.mean(EVI)]),
            'FOR': np.mean(FOR),
            'AVE': np.mean([np.mean(RES), np.mean(DEF), np.mean(EVI), np.mean(FOR)])
        }
        func.save_json(result, os.path.join(self.final_eval, f'{model}_final.json'))
    
    
    def iterate(self):
        for model in os.listdir(self.dialog_history_dir):
            processed_ids = []
            intermediate_model_folder = os.path.join(self.intermediate_eval,  model, 'DD')
            if not os.path.exists(intermediate_model_folder):
                os.makedirs(intermediate_model_folder)
            for file_name in os.listdir(intermediate_model_folder):
                if file_name.endswith('.json'):
                    case_id = file_name.split('-')[-1].replace('.json','')
                    processed_ids.append(int(case_id))

            self.dialog_history = func.load_jsonl(os.path.join(self.dialog_history_dir, model,  'DD_dialog_history.jsonl'))
            for dh in self.dialog_history:
                case_id = int(dh['case_id'].split('-')[-1].replace('.json',''))
                if case_id not in processed_ids:
                    for case in self.ground_truth:
                        if case_id == int(case['id'].split('-')[-1].replace('.json','')):
                            defendant = dh
                            break
                    print(f'Evaluating case {case_id}.')
                    defendant_type = 'personal' if 'gender' in case['specific_characters']['defendant'].keys() else 'company'
            
                    
                    self.evaluate(model, defendant, defendant_type, case)
                else:
                    print(f'Case {case_id} already evaluated.')
            self.get_final_score(intermediate_model_folder, model)
            
            

def main():
    parser = argparse.ArgumentParser()
    DDEvaluator.add_parser_args(parser)
    args = parser.parse_args()
    evaluator = DDEvaluator(args)
    evaluator.iterate()
    
if __name__ == '__main__':
    main()