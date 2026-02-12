import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.utils_func as func
from tools import law_tool
from tools import crime_tool
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import json

# 判断某个法条是否适用于该案件
async def fact_law_relevance_check(fact, law, bge_model, tool_calls_semaphore):
    # 给定法条名称获得法条内容
    law_content = law_tool.law_check(law, bge_model)
    prompt = f"""
    你是一名法律专家。请从四要件的角度分析某个法条是否适用于当前案件。
    法条：{law_content}
    案件：{fact}
    """
    result = (await func.get_completion_extract(tool_calls_semaphore, prompt, [], flag=0))[0]
    return result

# 对于法条检索时采用的query进行优化，结合案件信息，丰富query的内容
async def law_query_rewrite(query, context, tool_calls_semaphore):
    prompt = f"""
你是一名资深法律专家，擅长将用户的自然语言问题改写为更适合法条检索的专业查询语句。
【任务要求】
1. 结合给定的案件背景信息，对用户的原始查询进行改写；
2. 使查询更加明确、具体、符合法律表达习惯，便于检索相关法条；
3. 不要改变用户问题的核心法律含义；
4. 仅返回改写后的查询语句，不要输出分析过程。
【案件背景】
{context}
【原始查询】
{query}
【示例】
原始查询：重整计划执行阶段是否适用集中管辖？  
改写后：人民法院受理破产申请后，在重整计划执行阶段，若公司参与民事诉讼，是否需要遵循集中管辖的规定？
请给出改写后的查询：
"""
    result = (await func.get_completion_extract(tool_calls_semaphore, prompt, [], flag=0))[0]
    return result.strip()



def extract_law_articles(path):
    articles = []
    current_article = []

    article_start_pattern = re.compile(r'^(第[一二三四五六七八九十百千零\d]+条)')
    chapter_or_section_pattern = re.compile(r'^第[一二三四五六七八九十百千]+[章节]')

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if chapter_or_section_pattern.match(line):
                continue
            if article_start_pattern.match(line):
                if current_article:
                    articles.append("\n".join(current_article))
                    current_article = []
                current_article.append(line)
            else:
                current_article.append(line)
        if current_article:
            articles.append("\n".join(current_article))
    return articles

# 建立罪名-法条映射表
def build_crime_law_dict():
    articles = extract_law_articles('./src/data/knowledge/crime/laws.txt')
    crime_law_dict = {}

    crime_pattern = re.compile(r'【(.*?)】')
    article_number_pattern = re.compile(r'^(第[一二三四五六七八九十百千零\d]+条)')

    for article in articles:
        crime_match = crime_pattern.search(article)
        if not crime_match:
            continue
        crime_name = crime_match.group(1).strip()

        article_number_match = article_number_pattern.search(article)
        if article_number_match:
            article_number = article_number_match.group(1).strip()
        else:
            article_number = ""

        content = crime_pattern.sub("", article).strip()

        crime_law_dict[crime_name] = {
            "article_number": article_number,
            "content": content
        }
    with open('./src/data/knowledge/check/crime_law_dict.json','w',encoding='utf-8') as f:
        json.dump(crime_law_dict, f, ensure_ascii=False, indent=2)
    with open('./src/data/knowledge/check/crime_list.json','w',encoding='utf-8') as f:
        json.dump(list(crime_law_dict.keys()),f,ensure_ascii=False,indent=2)

def encode_all_crimes(bge_model):
    with open('./src/data/knowledge/check/crime_list.json','r',encoding='utf-8') as f:
        crime_list = json.load(f)
    embeddings = bge_model.encode(crime_list)
    np.save('./src/data/knowledge/check/crime_encodings.npy', embeddings)

# 输入罪名和法条名称（第*条），验证是否对应
def crime_law_consistency_check(crime,law,bge_model):
    with open('./src/data/knowledge/check/crime_encodings.npy','rb') as f:
        embeddings = np.load(f)
    with open('./src/data/knowledge/check/crime_list.json','r',encoding='utf-8') as f:
        crime_list = json.load(f)
    with open('./src/data/knowledge/check/crime_law_dict.json','r',encoding='utf-8') as f:
        crime_law_dict = json.load(f)

    format_crime = crime_tool.bge_match(crime,bge_model,embeddings,crime_list,1)[0]
    article_number = crime_law_dict[format_crime]["article_number"]
    content = crime_law_dict[format_crime]["content"]

    law = law_tool.extract_and_convert_article(law)
    result = ""
    if law == article_number:
        result += f"罪名“{crime}”和刑法{law}满足对应关系。"
        return result
    else:
        result += f"罪名“{crime}”和刑法{law}不满足对应关系，正确的法条是{content}。\n"
        return result
    

if __name__ == "__main__":
    # build_crime_law_dict()
    bge_model = SentenceTransformer("model/BAAI_bge-m3")
    # encode_all_crimes(bge_model)
    print(crime_law_consistency_check("武装叛乱","第104条",bge_model))
    print(crime_law_consistency_check("武装叛乱","第一百零四条",bge_model))
    print(crime_law_consistency_check("武装叛乱","第103条",bge_model))
    print(crime_law_consistency_check("抢劫罪","第103条",bge_model))
