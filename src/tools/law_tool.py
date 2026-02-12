import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from modelscope import AutoTokenizer, AutoModel
import torch

# 加载55347部法条
law_repository={}
with open("./src/data/knowledge/law/law.json", 'r', encoding='utf-8') as f:
    law_repository=json.load(f)

law_list = []
for k,v in law_repository.items():
    new_law = k + ":" + v
    law_list.append(new_law)
# print(law_list[0])

law_name_list = list(law_repository.keys())

# 阿拉伯数字到汉字数字的映射表
num_to_chinese = {
    0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
    6: "六", 7: "七", 8: "八", 9: "九", 10: "十", 100: "百"
}

def arabic_to_chinese(number):
    """将阿拉伯数字转换为中文数字（支持0-999）"""
    if not isinstance(number, int):
        # 如果是字符串，尝试转换为整数
        try:
            number = int(number)
        except ValueError:
            return str(number)
    
    if number < 0 or number > 999:
        return str(number)
    
    if number == 0:
        return "零"
    
    # 处理百位
    result = ""
    if number >= 100:
        hundreds = number // 100
        result += num_to_chinese[hundreds] + "百"
        number %= 100
        if number == 0:
            return result
    
    # 处理十位
    if number >= 10:
        tens = number // 10
        if tens > 0:
            if tens > 1 or (result and tens == 1):
                result += num_to_chinese[tens]
            result += "十"
        number %= 10
    elif result and number > 0:
        result += "零"
    
    # 处理个位
    if number > 0:
        result += num_to_chinese[number]
    
    return result

def extract_and_convert_article(text):
    """
    提取text中'第'和'条'之间的内容
    如果是中文数字则返回原text，否则将阿拉伯数字转换为中文数字后返回完整text
    """
    if "第" not in text or "条" not in text:
        return text
    
    # 找到"第"和"条"的位置
    start_index = text.find("第") + 1
    end_index = text.find("条", start_index)
    
    if start_index >= end_index:
        return text
    
    content = text[start_index:end_index].strip()
    
    if not content:
        return text
    
    # 判断是否是中文数字
    def is_chinese_number(s):
        """判断字符串是否全部由中文数字字符组成"""
        chinese_digits = set("零一二三四五六七八九十百")
        return all(char in chinese_digits for char in s)
    
    if is_chinese_number(content):
        return text

    try:
        arabic_num = int(content)
        chinese_num = arabic_to_chinese(arabic_num)
        
        new_text = text[:start_index] + chinese_num + text[end_index:]
        return new_text
    except ValueError:
        return text

def generate_bge_embeddings(input_list,output_path,bge_model):
    embeddings = bge_model.encode(input_list)
    np.save(output_path, embeddings)

def generate_sailer_embeddings(laws,output_path,sailer_model,sailer_tokenizer):
    law_embeddings = []
    for law in laws:
        inputs = sailer_tokenizer(law, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = sailer_model(**inputs)
            # 获取[CLS] token的嵌入向量，通常作为句子的向量表示
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        law_embeddings.append(embedding)
    np.save(output_path, law_embeddings)

# ---- 全局缓存，避免每次调用都从磁盘加载 ----
LAW_EMBEDDINGS = None
LAW_NAME_EMBEDDINGS = None


def _get_law_embeddings():
    global LAW_EMBEDDINGS
    if LAW_EMBEDDINGS is None:
        with open('./src/data/knowledge/law/law_encodings.npy','rb') as f:
            # 使用内存映射可降低内存峰值，且加载更快
            LAW_EMBEDDINGS = np.load(f)
    return LAW_EMBEDDINGS


def _get_law_name_embeddings():
    global LAW_NAME_EMBEDDINGS
    if LAW_NAME_EMBEDDINGS is None:
        with open('./src/data/knowledge/law/law_name_encodings.npy','rb') as f:
            LAW_NAME_EMBEDDINGS = np.load(f)
    return LAW_NAME_EMBEDDINGS


def bge_match(bge_model,query,embeddings,content_ls,topk):
    new_embedding = bge_model.encode([query])
    similarities = cosine_similarity(new_embedding,embeddings)
    topk_indices = np.argsort(similarities[0])[-topk:][::-1]
    topk_result = [content_ls[idx] for idx in topk_indices]
    return topk_result

def sailer_match(sailer_model,sailer_tokenizer,query,embeddings,content_ls,topk):
    inputs = sailer_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = sailer_model(**inputs)
        # 获取[CLS] token的嵌入向量，通常作为句子的向量表示
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    similarities = cosine_similarity(new_embedding,embeddings)
    topk_indices = np.argsort(similarities[0])[-topk:][::-1]
    topk_result = [content_ls[idx] for idx in topk_indices]
    return topk_result

# 给定法条名称，获得法条内容
def law_check(law_name,bge_model):
    # 规范输入的法条名称: 中华人民共和国*法第*条
    if  "中华人民共和国" not in law_name:
        law_name = "中华人民共和国" + law_name
    # print(law_name)
    law_name = extract_and_convert_article(law_name)
    # print(law_name)
    # 采用bge编码匹配key
    with open('./src/data/knowledge/law/law_name_encodings.npy','rb') as f:
        law_name_encodings = np.load(f)
    format_law_name = bge_match(bge_model,law_name,law_name_encodings,law_name_list,1)[0]
    # print(format_law_name)
    if format_law_name in list(law_repository.keys()):
        return format_law_name + ":" + law_repository[format_law_name]
    return law_name

# 给定一个query，获取top-k相关的法条名称及内容
# 注意介绍这个工具的时候和law_check作区分，如果仅需要获取某一个特定法条的内容，直接在字典中查找即可
def law_retrieval(query,topk,bge_model):
    # 加载law embeddings
    with open('./src/data/knowledge/law/law_encodings.npy','rb') as f:
        law_embeddings = np.load(f)
    topk_result = bge_match(bge_model,query,law_embeddings,law_list,topk)
    topk_result = '\n'.join(topk_result)
    return topk_result

def law_retrieval_sailer(query,topk,sailer_model,sailer_tokenizer):
    # 加载law embeddings
    with open('./src/data/knowledge/law/law_encodings_sailer.npy','rb') as f:
        law_embeddings = np.load(f)
    topk_result = sailer_match(sailer_model,sailer_tokenizer,query,law_embeddings,law_list,topk)
    topk_result = '\n'.join(topk_result)
    return topk_result

# 给定法条名称，如“刑法第*条”，首先获得该法条的具体内容，然后编码匹配
def law_recommendation(law,bge_model):
    law_content = law_check(law,bge_model)
    final_result = f"与{law}相似的法条有：\n"
    # 语义相似
    topk_result = law_retrieval(law_content,5,bge_model)
    final_result += topk_result
    final_result += '\n'
    # 结构相似
    index = law_content.find(":")
    if index != -1:
        law_name = law_content[:index]
        loc = law_name_list.index(law_name)
        # 计算前后索引，考虑边界情况
        start_index = max(0, loc - 2)  
        end_index = min(len(law_name_list), loc + 3) 
        
        surrounding_laws = []
        for i in range(start_index, end_index):
            if i != loc:  # 排除当前法条本身
                law_key = law_name_list[i]
                law_value = law_repository[law_key]
                surrounding_laws.append(f"{law_key}:{law_value}")
        
        # 去除重复法条
        if surrounding_laws:
            semantic_law_names = set()
            lines = topk_result.strip().split('\n')
            for line in lines:
                if ':' in line:
                    semantic_law_names.add(line.split(':')[0].strip())
            
            unique_surrounding_laws = []
            for law_text in surrounding_laws:
                law_key = law_text.split(':')[0]
                if law_key not in semantic_law_names:
                    unique_surrounding_laws.append(law_text)
            
            if unique_surrounding_laws:
                # final_result += "结构相似的法条：\n"
                final_result += "\n".join(unique_surrounding_laws)
                final_result += "\n"
    return final_result

if __name__ == "__main__":
    bge_model = SentenceTransformer("model/BAAI_bge-m3")
    # generate_bge_embeddings(law_list,'./src/data/knowledge/law/law_encodings.npy',bge_model)
    # generate_bge_embeddings(law_name_list,'./src/data/knowledge/law/law_name_encodings.npy',bge_model)
    
    # print(extract_and_convert_article('刑法第34条'))
    # print(extract_and_convert_article('刑法第二百零一条'))
    # print(law_retrieval('刑法第201条',3,bge_model))
    # print(answer = law_check('民法典第463条',bge_model))
    # print(law_recommendation('刑法第二百零一条',bge_model))
    # print(law_recommendation('民法典第463条',bge_model))
    print(law_retrieval('继承程序开始时间',5,bge_model))

    # 采用sailer
    # 加载SAILER_zh模型和分词器
    # sailer_tokenizer = AutoTokenizer.from_pretrained("model/SAILER_zh")
    # sailer_model = AutoModel.from_pretrained("model/SAILER_zh")
    # generate_sailer_embeddings(law_list,'./src/data/knowledge/law/law_encodings_sailer.npy',sailer_model,sailer_tokenizer)
    # print(law_retrieval_sailer('人民法院应当告知当事人对合议庭组成人员的什么权利',3,sailer_model,sailer_tokenizer))

