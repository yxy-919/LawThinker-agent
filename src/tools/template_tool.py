import re
def template_retrieval(template_type):
    if template_type == "起诉状":
        template = """
                            民事起诉状
            原告（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
            原告（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
            
            被告（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
            被告（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。

            诉讼请求：
            1. 诉讼请求1.
            2. 诉讼请求2.
            ...
            事实和理由：
            ...
            证据和证据来源，证人姓名和住所：
            ...
        """
        instruction = """
        注意，如果原告或被告是自然人，则需要获取具体的姓名、性别、出生日期、民族和住址。如果原告或被告是法人，则需要获取具体的名称、法定代表人、住所。
        """
        context = f"模版如下：\n{template}\n使用模版时请注意：\n{instruction}"
        return context
     
    elif template_type == "答辩状":
        template = """
                            民事答辩状
            答辩人（如果是自然人）：XXX，男/女，XXXX年XX月XX日生，X族，住XXXXXX。
            答辩人（如果是法人）：XXX，法定代表人：XXX，住所：XXXXXX。
            
            对XXXX人民法院（XXXX）...民初...号...（写明当事人和案由）一案的起诉，答辩如下：
            ......（写明答辩意见）
            证据和证据来源，证人姓名和住所：
            ......
        """
        instruction = """
        注意，如果答辩人是自然人，则需要获取具体的姓名、性别、出生日期、民族和住址。如果答辩人是法人，则需要获取具体的名称、法定代表人、住所。
        """
        context = f"模版如下：\n{template}\n使用模版时请注意：\n{instruction}"
        return context
    
    else:
        return "请输入正确的文书类型：起诉状或答辩状。"

def format_item_names(name_set, correct_name, model_complaint):
    for item in name_set:
        model_complaint = model_complaint.replace(item, correct_name)
    return model_complaint


def document_format_check(document_type,document):
    if document_type == "起诉状":
        check_result = "文书的检查结果如下：\n"
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
        reform_document = format_item_names(incorrect_plaintiff_names, '原告：', document)
        reform_document = format_item_names(incorrect_defendant_names, '被告：', reform_document)
        reform_document = format_item_names(['证据和证据来源：'], '证据和证据来源，证人姓名和住所：', reform_document)
        # 检查完整性
        # 对各个部分单独检查

        flag = 1
        # keys = ['原告：', '被告：', '诉讼请求：', '事实和理由：', '证据和证据来源，证人姓名和住所：']
        # for key in keys:
        #     if key not in reform_document:
        #         check_result += f"缺乏“{key[:-1]}”部分\n"
        #         flag = 0
        for key in incorrect_plaintiff_names:
            if key in document:
                check_result += f"{key}的格式有误，正确的格式应该是：“原告：”。\n"
                flag = 0
                break
        for key in incorrect_defendant_names:
            if key in document:
                check_result += f"{key}的格式有误，正确的格式应该是：“被告：”。\n"
                flag = 0
                break
        if '诉讼请求：' not in document:
            check_result += "缺乏“诉讼请求：”部分\n"
            flag = 0
        if '事实和理由：' not in document:
            check_result += "缺乏“事实和理由：”部分\n"
            flag = 0
        if '证据和证据来源：' in document:
            check_result += "“证据和证据来源：”的格式有误，正确的格式应该是：“证据和证据来源，证人姓名和住所：”。\n"
            flag = 0

        if flag == 0:
            check_result += "请按照正确的格式进行修改。\n"
            # return check_result            
        if "X" in document:
            check_result += "请向用户确认具体的信息代替“X”。\n"
        if "男" in document or "女" in document or "族" in document or "年" in document:
            # 是自然人
            if "男" not in document or "女" not in document:
                check_result += "请向用户确认具体的性别。\n"
            if "族" not in document:
                check_result += "请向用户确认具体的民族。\n"
            if "年" not in document:
                check_result += "请向用户确认具体的出生年份。\n"
        # 检查顺序正确
        # else:
            # positions = {}
            # for key in keys:
            #     pos = reform_document.find(key)
            #     positions[key] = pos
            # sorted_by_pos = sorted(keys, key=lambda k: (positions[k] if positions[k] is not None else float('inf')))
            # if sorted_by_pos == keys:
            #     check_result += "文书的格式正确。\n"
            #     return check_result
            # check_result += "文书中各个部分的顺序有误，正确的顺序应该是：原告的描述、被告的描述、诉讼请求部分、事实和理由部分、证据和证据来源，证人姓名和住所部分。请按照正确的顺序进行修改。\n"
            # return check_result
        last_pos = -1
        sequential_score = -1
        for char in ['原告：', '被告：', '诉讼请求：', '事实和理由：', '证据和证据来源，证人姓名和住所：']:
            if char not in reform_document:
                sequential_score = 0
            current_pos = reform_document.find(char, last_pos + 1)
            if current_pos == -1:
                sequential_score = 0
            last_pos = current_pos
        if sequential_score == 0:
            check_result += "文书中各个部分的顺序有误，正确的顺序应该是：原告的描述、被告的描述、诉讼请求部分、事实和理由部分、证据和证据来源，证人姓名和住所部分。请按照正确的顺序进行修改。\n"
        return check_result
        
    elif document_type == "答辩状":
        check_result = "文书的检查结果如下：\n"
        incorrect_defendant_names = [
            '答辩人（自然人）：',
            '答辩人（自然人或法人）：',
            '答辩人（如果是自然人）：',
            '答辩人（如果是法人）：',
            '答辩人（法人）：'
            ]
        reform_document = format_item_names(incorrect_defendant_names, '答辩人：', document)
        # 检查完整性
        flag= 1
        # keys = ['答辩人：', '答辩如下：', '证据和证据来源，证人姓名和住所：']
        # for key in keys:
        #     if key not in reform_document:
        #         check_result += f"缺乏“{key[:-1]}”部分\n"
        #         flag = 0
        for key in incorrect_defendant_names:
            if key in document:
                check_result += f"{key}的格式有误，正确的格式应该是：“答辩人：”。\n"
                flag = 0
                break

        defense_pattern = r'对(.*?)人民法院（(.*?)）(.*?)民初(.*?)号(.*?)一案的起诉，答辩如下：'
        defense = re.findall(defense_pattern, document)
        if defense == []:
            check_result += "答辩部分的格式有误，正确的格式应该是：“对**人民法院**民初**号**一案的起诉，答辩如下：”，其中**必须根据案件实际内容替换为具体信息，其余所有文字必须保持完全不变。\n"
            flag = 0

        if '证据和证据来源：' in document:
            check_result += "“证据和证据来源：”的格式有误，正确的格式应该是：“证据和证据来源，证人姓名和住所：”。\n"
            flag = 0

        if flag == 0:
            check_result += "请按照正确的格式进行修改。\n"
            # return check_result            
        if "X" in document:
            check_result += "请向用户确认具体的信息代替“X”。\n"
        if "男" in document or "女" in document or "族" in document or "年" in document:
            # 是自然人
            if "男" not in document or "女" not in document:
                check_result += "请向用户确认具体的性别。\n"
            if "族" not in document:
                check_result += "请向用户确认具体的民族。\n"
            if "年" not in document:
                check_result += "请向用户确认具体的出生年份。\n"
        # 检查顺序正确
        # else:
        #     positions = {}
        #     for key in keys:
        #         pos = reform_document.find(key)
        #         positions[key] = pos
        #     sorted_by_pos = sorted(keys, key=lambda k: (positions[k] if positions[k] is not None else float('inf')))
        #     if sorted_by_pos == keys:
        #         check_result += "文书的格式正确。\n"
        #         return check_result
        #     check_result += "文书中各个部分的顺序有误，正确的顺序应该是：答辩人的描述、答辩如下部分、证据和证据来源，证人姓名和住所部分。请按照正确的顺序进行修改。\n"
        #     return check_result
        last_pos = -1
        sequential_score = -1
        for char in ['答辩人：', '答辩如下：', '证据和证据来源，证人姓名和住所：']:
            if char not in reform_document:
                sequential_score = 0
            current_pos = reform_document.find(char, last_pos + 1)
            if current_pos == -1:
                sequential_score = 0
            last_pos = current_pos
        if sequential_score == 0:
            check_result += "文书中各个部分的顺序有误，正确的顺序应该是：答辩人的描述、答辩如下部分、证据和证据来源，证人姓名和住所部分。请按照正确的顺序进行修改。\n"
        return check_result
   
    else:
        return "请输入正确的文书类型：起诉状或答辩状。"

if __name__ == "__main__":
    # print(template_retrieval("答辩状"))
    # doc = ""
    print(document_format_check("答辩状",doc))