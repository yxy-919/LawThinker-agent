import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.utils_func as func

def plan_generation(document_type):
    if document_type == "起诉状":
        return """
        对于民事起诉状，需要收集以下信息：
        （1）如果原告是自然人，则需要收集原告的姓名、性别、出生日期、民族、住址等个人信息。
        （2）如果原告是法人，则需要收集原告的名称、法定代表人、住所等个人信息。
        （3）如果被告是自然人，则需要收集被告的姓名、性别、出生日期、民族、住址等个人信息。
        （4）如果被告是法人，则需要收集被告的名称、法定代表人、住所等个人信息。
        （5）收集原告的诉讼请求。
        （6）收集原告的事实和理由。
        （7）收集原告的证据和证据来源，证人姓名和住所。
        """
    elif document_type == "答辩状":
        return """
        对于民事答辩状，需要收集以下信息：
        （1）如果答辩人是自然人，则需要收集答辩人的姓名、性别、出生日期、民族、住址等个人信息。
        （2）如果答辩人是法人，则需要收集答辩人的名称、法定代表人、住所等个人信息。
        （3）收集答辩人的答辩意见。
        （4）收集答辩人的证据和证据来源，证人姓名和住所。
        """
    else:
        return "请输入正确的文书类型：起诉状或答辩状。"
        
# def plan_generation(document_type,template):
#     if template is not None:
#         prompt = f"""
#         你是一名法律专家。请根据以下文书的类型以及给定的模板，生成一份完成该文书所需要收集信息的详细计划。
#         注意：全面地收集与该文书相关的所有信息。

#         文书类型：{document_type}
#         模板：{template}
#         请按照如下的步骤输出计划：
#         （1）……
#         （2）……
#         ……
#         """
#         result = func.get_completion_extract(prompt,[],flag = 0)[0]
#         return result
#     else:
#         return "没有提供模板，无法生成计划。请先调用template_retrieval工具获取模板。"
    