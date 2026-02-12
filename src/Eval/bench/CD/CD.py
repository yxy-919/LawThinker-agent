import re
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(parent_dir)
import utils.utils_func as func
import argparse

class CDEvaluator:
    def __init__(self, args):
        self.ground_truth = func.load_jsonl('./src/data/case/J1-Eval_CD.jsonl')
        self.args = args
        
        self.dialog_history_dir = args.dialog_history_dir
        self.intermediate_eval = args.intermediate_eval
        self.final_eval = args.final_eval
        
        
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument("--dialog_history_dir", default = "./src/data/dialog_history", type = str)
        parser.add_argument("--intermediate_eval", default = "./src/Eval/eval_result", type = str)
        parser.add_argument("--final_eval", default = "./src/Eval/final_result/CD", type = str)    
    
    def _clean(self, text):
        clean_words = ['。', '法定代表人：', '住所：', '住', ' ']
        for word in clean_words:
            text = text.replace(word, '')
        return text
    
    def validate_ground_truth_items(self, ground_truth, key):
        return ground_truth[key] != None
    
    def reformat(self, text, plaintiff_type, defendant_type, ground_truth):
        # 提取原告信息
        plaintiff_patterns = [
            r"原告（自然人）：(.*?)\n\n",
            r"原告（自然人或法人）：(.*?)\n\n",
            r"原告（如果是自然人）：(.*?)\n\n",
            r"原告（如果是法人）：(.*?)\n\n",
            r"原告（法人）：(.*?)\n\n",
            r"原告：(.*?)\n\n被告",
            r"原告：(.*?)\n被告",
            r"\*\*原告\*\*(.*?)\n\n\*\*被告\*\*"
        ]
        plaintiff_info = None
        for pattern in plaintiff_patterns:
            match = re.search(pattern, text.replace(' ','').replace('*',''), re.DOTALL)
            if match:
                plaintiff_info = match.group(1).strip()
                break
        plaintiff = plaintiff_info if plaintiff_info else "未找到原告信息"
        if plaintiff == '未找到原告信息':
            plaintiff = text
            
        # 提取被告信息
        defendant_patterns = [
            r"被告（自然人）：(.*?)\n\n",
            r"被告（自然人或法人）：(.*?)\n\n",
            r"被告（如果是自然人）：(.*?)\n\n",
            r"被告（如果是法人）：(.*?)\n\n",
            r"被告（法人）：(.*?)\n\n",
            r"被告：(.*?)\n\n诉讼",
            r"被告：(.*?)\n\n\*\*诉讼",
            r"被告：(.*?)\n\n诉讼请求：",
            r"\*\*被告\*\*(.*?)\n\n\*\*诉讼请求\*\*"
        ]
        defendant_info = None
        for pattern in defendant_patterns:
            match = re.search(pattern, text.replace(' ','').replace('*',''), re.DOTALL)
            if match:
                defendant_info = match.group(1).strip()
                break
        defendant = defendant_info if defendant_info else "未找到被告信息"
        if defendant == '未找到被告信息':
            defendant = text
        
        temp = text.replace('*','').replace(' ','')
        # 提取诉讼请求
        claims_patterns = [
            r"诉讼请求：\n(.*?)\n\n事实与理由",
            r"诉讼请求：\n(.*?)\n\n事实和理由",
            r"诉讼请求：\n(.*?)\n...\n事实和理由",
            r"\*\*诉讼请求：\*\*\n(.*?)\n\n\*\*"
        ]
        for claims_pattern in claims_patterns:
            claims_info = re.search(claims_pattern, temp, re.DOTALL)
            if claims_info:
                claims = claims_info.group(1).replace(' ','').split("\n")
                claims = [cl for cl in claims if len(cl) > 0]
                break
            else:
                claims = '未找到诉讼请求'
        if claims == '未找到诉讼请求':
            claims = text
        
        
        # 提取事实和理由
        facts_patterns = [
            r"事实和理由：\n(.*?)\n\n",
            r"事实与理由：(.*?)\n\n证据",
            r"\*\*事实与理由：\*\*\n(.*?)\n\n\*\*",
            r"事实与理由：\n(.*?)\n\n此致",
            r"事实和理由：\n(.*?)\n\n此致"
        ]
        for facts_pattern in facts_patterns:
            facts_info = re.search(facts_pattern, temp.replace('*',''), re.DOTALL)
            if facts_info:
                facts = facts_info.group(1).replace(' ','')
                break
            else:
                facts = "未找到事实和理由"
        if facts == '未找到事实和理由':
            facts = text        
        

        # 提取证据
        evidence_patterns = [
            r"证据和证据来源，证人姓名和住所：(.*?)\n\n此致",
            r"证据和证据来源、证人姓名和住所：(.*?)\n\n此致",
            r"证据和证据来源，证人姓名和住所：(.*?)\n\n<询问结束>",
            r"证据和证据来源：(.*?)\n\n<询问结束>",
            r"证据和证据来源：(.*?)\n\n此致",
            r"\*\*证据和证据来源、证人姓名和住所：\*\*\n(.*?)\n\n此致",
            r"证据和证据来源，证人姓名和住所：\n(.*?)。",
            r"\*\*证据和证据来源：\*\*\n(.*?)\n\n<询问结束>"
        ]
        evidence_info = None
        for evidence_pattern in evidence_patterns:
            evidence_info = re.search(evidence_pattern, temp.replace(' ','').replace('*',''), re.DOTALL)
            if evidence_info:
                evidence = evidence_info.group(1).replace(' ','').split("\n")
                evidence = [ev for ev in evidence if len(ev) > 0]
                break
            else:
                evidence = ["未找到证据"]
        if len(evidence) == 0:
            evidence = ['未找到证据']
        if evidence[0] == '未找到证据':
            evidence = [text]
        
        model_answer = {
            "plaintiff": plaintiff,
            "defendant": defendant
        }
        model_answer['claims'] = claims
        model_answer['facts'] = facts
        model_answer['evidence'] = evidence
        return model_answer
    
    def compare_plaintiff_defendant(self, plaintiff_type, defendant_type, ground_truth, model_answer):
        plaintiff = ground_truth['specific_characters']['plaintiff']
        defendant = ground_truth['specific_characters']['defendant']
        plaintiff_claim = ground_truth['claims']
        plaintiff_case_details = ground_truth['case_details']
        plaintiff_evidence = ground_truth['evidence']
        
        # 统计原告
        model_plaintiff = model_answer['plaintiff']
        pla = 0
        t_score = 0
        if plaintiff_type == 'personal':
            gt_name = plaintiff['name']
            gt_sex = plaintiff['gender']
            gt_birth = plaintiff['birth_date']
            gt_address = plaintiff['address']
            gt_ethnicity = plaintiff['ethnicity']
            if gt_name:
                if gt_name in model_plaintiff:
                    pla += 1
                t_score += 1
            if gt_sex:
                if gt_sex in model_plaintiff:
                    pla += 1
                t_score += 1
            if gt_birth:
                if gt_birth.replace('X','x') in model_plaintiff.replace('X','x'):
                    pla += 1
                t_score += 1
            if gt_address:
                if gt_address in model_plaintiff:
                    pla += 1
                t_score += 1
            if gt_ethnicity:
                if gt_ethnicity in model_plaintiff:
                    pla += 1
                t_score += 1
        else:
            gt_name = plaintiff['name']
            gt_address = plaintiff['address']
            gt_representative = plaintiff['representative']
            if gt_name:
                if gt_name in model_plaintiff:
                    pla += 1
                t_score += 1
            if gt_address:
                if gt_address in model_plaintiff:
                    pla += 1
                t_score += 1
            if gt_representative:
                if gt_representative in model_plaintiff:
                    pla += 1
                t_score += 1
        #if 'xx年' in model_plaintiff or 'xxxx' in model_plaintiff or 'XXXX' in model_plaintiff:
        #    pla = 0
        pla_score = pla / t_score
        
        # 统计被告
        model_defendant = model_answer['defendant']
        defen = 0
        t_score = 0
        if defendant_type == 'personal':
            gt_name = defendant['name']
            gt_sex = defendant['gender']
            gt_birth = defendant['birth_date']
            gt_address = defendant['address']
            gt_ethnicity = defendant['ethnicity']
            if gt_name:
                if gt_name in model_defendant:
                    defen += 1
                t_score += 1
            if gt_sex:
                if gt_sex in model_defendant:
                    defen += 1
                t_score += 1
            if gt_birth:
                if gt_birth.replace('X','x') in model_defendant.replace('X','x'):
                    defen += 1
                t_score += 1
            if gt_address:
                if gt_address in model_defendant:
                    defen += 1
                t_score += 1
            if gt_ethnicity:
                if gt_ethnicity in model_defendant:
                    defen += 1
                t_score += 1
        else:
            gt_name = defendant['name']
            gt_address = defendant['address'].replace('（','').replace('）','')
            gt_representative = defendant['representative']
            if gt_name:
                if gt_name in model_defendant:
                    defen += 1
                t_score += 1
            if gt_address:
                if gt_address in model_defendant:
                    defen += 1
                t_score += 1
            if gt_representative:
                if gt_representative in model_defendant:
                    defen += 1
                t_score += 1
        defen_score = defen / t_score
        
        prompt_cla = '''你是一名法律专家。请根据“法律问题”和“标准答案”，判断“待评测答案”是否完全、准确地涵盖了“标准答案”的所有核心要点，由此给出0-10分的评分。你**不用考虑待评测答案表达是否简洁、重点是否突出、是否使用寒暄语、结构是否冗长等非实质性因素**，无须因不够简洁而扣分。你只需要输出分数，不要输出理由。

        标准答案：
        {gt_answer}

        待测评答案：
        {model_answer}
        '''
        
        model_claims = '\n'.join(model_answer['claims'])
        cla_scores = 0
        total_cla_scores = 0
        full_clas = ''
        count = 1
        for gtc in plaintiff_claim:
            full_clas += f'{count}. {gtc}\n'
            count += 1
            
        if model_answer['claims'] == '未找到诉讼请求':
                temp = 0
        else:
            full_prompt_cla = prompt_cla.format(gt_answer=full_clas, model_answer=model_claims)
            temp = float(func.get_completion(full_prompt_cla, [], flag=0)[0].replace('分',''))/10
        cla_scores += temp
        total_cla_scores += 1
        cla_score = cla_scores/total_cla_scores
        
        prompt_evi = '''你是一名法律专家。请根据“法律问题”和“标准答案”，判断“待评测答案”是否完全、准确地涵盖了“标准答案”的所有核心要点，由此给出0-10分的评分。你**不用考虑待评测答案表达是否简洁、重点是否突出、是否使用寒暄语、结构是否冗长等非实质性因素**，无须因不够简洁而扣分。你只需要输出分数，不要输出理由。

        标准答案：
        {gt_answer}

        待测评答案：
        {model_answer}
        '''
        
        model_evidence = '\n'.join(model_answer['evidence'])
        gt_evidences = []
        for pe in plaintiff_evidence.keys():
            gt_evidences.extend(plaintiff_evidence[pe]['evidence'].replace('\n', '、').split('、'))
        evi_scores = 0
        total_evi_scores = 0
        
        count = 1
        full_evis = ''
        for gte in gt_evidences:
            full_evis += f'{count}. {gte}\n'
            count += 1
            
        if model_evidence == '未找到证据' or '（待补充）' in model_evidence or '...(待补充)' in model_evidence or  '...' in model_evidence or '（根据您后续提供的证据信息补充）' in model_evidence or '（此处留空待补充）' in model_evidence:
            temp = 0
        else:
            full_prompt_evi = prompt_evi.format(gt_answer=full_evis, model_answer=model_evidence)
            temp = float(func.get_completion(full_prompt_evi, [], flag = 0)[0].replace('分',''))/10
        evi_scores += temp
        total_evi_scores += 1
        evi_score = evi_scores/total_evi_scores if total_evi_scores > 0 else 0
        
        
        prompt_fac = '''你是一名法律专家。请根据“法律问题”和“标准答案”，判断“待评测答案”是否完全、准确地涵盖了“标准答案”的所有核心要点，由此给出0-10分的评分。你**不用考虑待评测答案表达是否简洁、重点是否突出、是否使用寒暄语、结构是否冗长等非实质性因素**，无须因不够简洁而扣分。你只需要输出分数，不要输出理由。

        标准答案：
        {gt_answer}

        待测评答案：
        {model_answer}
        '''
        model_facts = model_answer['facts']
        gt_facts = plaintiff_case_details
        full_prompt_fac = prompt_fac.format(gt_answer = gt_facts, model_answer = model_facts)
        if model_facts != '未找到事实和理由' and model_facts != '...' and model_facts != '（待补充）' and model_facts != '（根据您后续提供的事实信息补充）':
            temp = func.get_completion(full_prompt_fac, [], flag=0)[0].replace('分','')
            facts_score = float(temp)/10
        else:
            facts_score = 0
        
        accuracy = {
            "plaintiff": {'pla_score': pla_score, 'ground_truth': plaintiff, 'model_plaintiff': model_plaintiff},
            "defendant": {'defen_score':defen_score, 'ground_truth': defendant, 'model_defendant': model_defendant},
            "claims": {'cla_score': cla_score, 'ground_truth': plaintiff_claim, 'model_claims': model_claims},
            "evidences": {'evi_score':evi_score, 'ground_truth': gt_evidences, 'model_evidence': model_evidence},
            "facts_and_reasons": {'facts_score': facts_score, 'gt_facts': gt_facts, 'model_facts': model_facts},
            "DOC": np.mean([pla_score, defen_score, cla_score, evi_score, facts_score])
        }
        return accuracy
    
    def format_item_names(self, name_set, correct_name, model_complaint):
        for item in name_set:
            model_complaint = model_complaint.replace(item, correct_name)
        return model_complaint
    
    def format_following_score(self, model_complaint):
        # 名称正确性得分
        label_score = 0
        for i in ['原告：', '被告：', '诉讼请求：', '事实和理由：', '证据和证据来源，证人姓名和住所：']:
            if i in model_complaint:
                label_score += 1
        
        # 顺序得分
        incorrect_plaintiff_names = [
            '原告（自然人）：',
            '原告（自然人或法人）：',
            '原告（如果是自然人）：',
            '原告（如果是法人）：',
            '原告（法人）：',
        ]
        incorrect_defendant_names = [
            '被告（自然人）：',
            '被告（自然人或法人）：',
            '被告（如果是自然人）：',
            '被告（如果是法人）：',
            '被告（法人）：'
            ]
        reform_model_complaint = self.format_item_names(incorrect_plaintiff_names, '原告：', model_complaint)
        reform_model_complaint = self.format_item_names(incorrect_defendant_names, '被告：', reform_model_complaint)
        
        # 修改：有逻辑错误
        # last_pos = -1
        # for char in ['原告：', '被告：', '诉讼请求：', '事实和理由：', '证据和证据来源，证人姓名和住所：']:
        #     if char not in reform_model_complaint:
        #         sequential_score = 0
        #     current_pos = reform_model_complaint.find(char, last_pos + 1)
        #     if current_pos == -1:
        #         sequential_score = 0
        #     last_pos = current_pos
        # sequential_score = 1

        # 二次修改，reform_model_complaint之后再判断顺序
        # keys = ['原告：', '被告：', '诉讼请求：', '事实和理由：', '证据和证据来源，证人姓名和住所：']
        # positions = []

        # sequential_score = 1 

        # for key in keys:
        #     pos = model_complaint.find(key)
        #     if pos == -1:
        #         sequential_score = 0 
        #         break
        #     positions.append(pos)

        # # 如果五个字段都存在，再检查它们是否按升序排列
        # if sequential_score == 1:
        #     if not (positions[0] < positions[1] < positions[2] < positions[3] < positions[4]):
        #         sequential_score = 0    

        last_pos = -1
        sequential_score = -1
        for char in ['原告：', '被告：', '诉讼请求：', '事实和理由：', '证据和证据来源，证人姓名和住所：']:
            if char not in reform_model_complaint:
                sequential_score = 0
            current_pos = reform_model_complaint.find(char, last_pos + 1)
            if current_pos == -1:
                sequential_score = 0
            last_pos = current_pos
        if sequential_score == -1:
            sequential_score = 1

        FOR = sequential_score * (label_score/5)
        format_following_score = {
            "label_score": label_score,
            "sequential_score": sequential_score,
            "FOR": FOR,
            'model_complaint': model_complaint
        }
        
        return format_following_score
    
    def evaluate(self, complaint_dh, model, plaintiff_type, defendant_type, ground_truth):
        flag = False
        # 文书在最后生成
        for dialog in reversed(complaint_dh['dialog_history']):
            if dialog['role'] == 'Lawyer' and ('民事起诉状' in dialog['content'] or ('原告' in dialog['content'] and '被告' in dialog['content'] and '证据和证据来源' in dialog['content'])):
                model_complaint = dialog['content']
                flag = True
                break
        id = int(ground_truth['id'].split('-')[-1])
        
        if flag:
            model_answer = self.reformat(model_complaint, plaintiff_type, defendant_type, ground_truth)
            accuracy = self.compare_plaintiff_defendant(plaintiff_type, defendant_type, ground_truth, model_answer)
            format_following_score = self.format_following_score(model_complaint)
            evaluation_result = {
                "DOC": accuracy,
                "FOR": format_following_score,
                "AVE": np.mean([accuracy['DOC'], format_following_score['FOR']]),
                'dialog_history': complaint_dh['dialog_history']
            }
        else:
            print(f"Case {id} task unfinished!")
            evaluation_result = {
                'DOC': 0,
                'FOR': 0,
                'AVE': 0,
                'dialog_history': complaint_dh['dialog_history']
            }
        func.save_json(evaluation_result, os.path.join(self.intermediate_eval,  model, 'CD',  f'CD_{id}.json'))
    
    def get_final_score(self, intermediate_model_folder, model):
        PLA = []
        DEF = []
        CLA = []
        EVI = []
        FAC = []
        DOC = []
        FOR = []
        for file_name in os.listdir(intermediate_model_folder):
            data = func.load_json(os.path.join(intermediate_model_folder, file_name))
            if data['DOC'] != 0:
                pla_score = data['DOC']['plaintiff']['pla_score']
                def_score = data['DOC']['defendant']['defen_score']
                cla_score = data['DOC']['claims']['cla_score']
                evi_score = data['DOC']['evidences']['evi_score']
                fac_score = data['DOC']['facts_and_reasons']['facts_score']
            else:
                pla_score = 0
                def_score = 0
                cla_score = 0
                evi_score = 0
                fac_score = 0
            doc = np.mean([pla_score, def_score, cla_score, evi_score, fac_score])
            if data['FOR'] != 0:
                for_score = data['FOR']['FOR']
            else:
                for_score = 0
            PLA.append(pla_score)
            DEF.append(def_score)
            CLA.append(cla_score)
            EVI.append(evi_score)
            FAC.append(fac_score)
            DOC.append(doc)
            FOR.append(for_score)
        print('PLA', np.mean(PLA))
        print('DEF', np.mean(DEF))
        print('CLA', np.mean(CLA))
        print('EVI', np.mean(EVI))
        print('FAC', np.mean(FAC))
        print('DOC', np.mean(DOC))
        print('FOR', np.mean(FOR))
        result = {
            'model': model,
            'PLA': np.mean(PLA),
            'DEF': np.mean(DEF),
            'CLA': np.mean(CLA),
            'EVI': np.mean(EVI),
            'FAC': np.mean(FAC),
            'DOC': np.mean(DOC),
            'FOR': np.mean(FOR),
            'AVE': np.mean([np.mean(DOC), np.mean(FOR)]),
        }
        func.save_json(result, os.path.join(self.final_eval, f'{model}_final.json'))
    
    
    def iterate(self):
        for model in os.listdir(self.dialog_history_dir):
            evaluated_ids = []
            intermediate_model_folder = os.path.join(self.intermediate_eval,  model, 'CD')
            if not os.path.exists(intermediate_model_folder):
                os.makedirs(intermediate_model_folder)
            for file_name in os.listdir(intermediate_model_folder):
                if file_name.endswith('.json'):
                    evaluated_ids.append(int(file_name.split('_')[1].split('.')[0]))


            self.dialog_history = func.load_jsonl(os.path.join(self.dialog_history_dir, model,  'CD_dialog_history.jsonl'))
            for dh in self.dialog_history:
                case_id = int(dh['case_id'].split('-')[-1])
                if case_id not in evaluated_ids:
                    print(f'Evaluating case {case_id}')
                    for case in self.ground_truth:
                        if int(case['id'].split('-')[-1]) == case_id:
                            complaint = dh
                            break
                    plaintiff_type = 'personal' if 'gender' in case['specific_characters']['plaintiff'].keys() else 'company'
                    defendant_type = 'personal' if 'gender' in case['specific_characters']['defendant'].keys() else 'company'
                else:
                    print(f'Case {case_id} already evaluated, skipping.')
                    continue
                
                self.evaluate(complaint, model, plaintiff_type, defendant_type, case)
            self.get_final_score(intermediate_model_folder, model)
        

def main():
    parser = argparse.ArgumentParser()
    CDEvaluator.add_parser_args(parser)
    args = parser.parse_args()
    evaluator = CDEvaluator(args)
    evaluator.iterate()
    
if __name__ == '__main__':
    main()