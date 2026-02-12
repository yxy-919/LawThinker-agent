import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 按照刑法中的章节结构进行归类
def get_structure():
    result = {}
    with open('./src/data/knowledge/crime/laws.txt', 'r', encoding='utf-8') as f:
        for idx,line in enumerate(f):
            if line == '\n':
                continue
            chapter_pattern = re.compile(r'^第[一二三四五六七八九十]章\s+(.*)')
            section_pattern = re.compile(r'^第[一二三四五六七八九十]节\s+(.*)')
            chapter_match = chapter_pattern.match(line)
            section_match = section_pattern.match(line)
            if chapter_match:
                title = chapter_match.group(1)
                result[title] = []
            elif section_match:
                title = section_match.group(1)
                result[title] = []
            else:
                match = re.search(r'【(.*?)】',line)
                if match:
                    crime_content = match.group(1)
                    # 检查是否有分号
                    if ";" in crime_content:
                        split_crimes = [c.strip() for c in crime_content.split(";")]
                    else:
                        split_crimes = [crime_content.strip()]
                    # 检查每个罪名中是否包含两个“罪”字
                    for crime in split_crimes:
                        if crime.count("罪") != 1:
                            crimes = re.split(r'(?<=罪)、', crime)
                            result[title].extend(crimes)
                        else:
                            result[title].append(crime)

    crime_list = []
    for value in result.values():
        if len(value) > 0:
            crime_list.append(value)
    with open('./src/data/knowledge/crime/crime_list','w',encoding='utf-8') as f:
        json.dump(crime_list,f,ensure_ascii=False,indent=2)

def bge_match(charge,bge_model,embeddings,crime_names,topk):
    new_embedding = bge_model.encode([charge])
    similarities = cosine_similarity(new_embedding,embeddings)
    topk_indices = np.argsort(similarities[0])[-topk:][::-1]
    topk_charges = [crime_names[idx] for idx in topk_indices]
    return topk_charges

def encode_all_crimes(bge_model):
    with open('./src/data/knowledge/crime/crime_list','r',encoding='utf-8') as f:
        crime_list = json.load(f)
    crimes_names = []
    for chapter in crime_list:
        crimes_names.extend(chapter)
    embeddings = bge_model.encode(crimes_names)
    np.save('./src/data/knowledge/crime/crime_encodings.npy', embeddings)

def are_crimes_in_same_sublist(crime1,crime2,crime_list):
    """
    判断两个罪名是否位于同一个子列表中。

    :param crime1: 第一个罪名
    :param crime2: 第二个罪名
    :param crime_list: 包含多个子列表的罪名列表
    :return: 如果两个罪名在同一个子列表中，返回 True；否则返回 False
    """
    for sublist in crime_list:
        if crime1 in sublist and crime2 in sublist:
            return True
    return False

def charge_expansion(charges,bge_model):
    k = 3
    with open('./src/data/knowledge/crime/crime_encodings.npy','rb') as f:
        embeddings = np.load(f)
    # print(embeddings.shape)
    # (513, 1024)
    with open('./src/data/knowledge/crime/crime_list','r',encoding='utf-8') as f:
        crime_list = json.load(f)
    crime_names = []
    for chapter in crime_list:
        crime_names.extend(chapter)
    all_expanded_charges = []
    for charge in charges:
        topk_crimes = bge_match(charge,bge_model,embeddings,crime_names,topk=2*k+10)
        # print(topk_crimes)
        # 把topk的相似罪名划分成同一章节和不同章节
        same_chapter = []
        different_chapter = []
        expanded_crimes = []
        for crime in topk_crimes:
            if crime != charge:
                #判断crime和charge是否处于同一章节
                if are_crimes_in_same_sublist(charge,crime,crime_list):
                    same_chapter.append(crime)
                else:
                    different_chapter.append(crime)
        expanded_crimes.extend(same_chapter[:k])
        expanded_crimes.extend(different_chapter[:k])
        all_expanded_charges.extend(expanded_crimes)
        # print("同一章节罪名：",same_chapter[:k])
        # print("不同章节罪名：",different_chapter[:k])
    all_expanded_charges = list(set(all_expanded_charges))
    all_expanded_charges = "、".join(all_expanded_charges)
    return all_expanded_charges

if __name__ == "__main__":
    # get_structure()
    bge_model = SentenceTransformer("model/BAAI_bge-m3")
    encode_all_crimes(bge_model)
    print(charge_expansion(["故意伤害罪"],bge_model,3))