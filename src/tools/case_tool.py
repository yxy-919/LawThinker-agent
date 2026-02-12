import os
import json
import numpy as np
import faiss
from transformers import AutoModel, AutoTokenizer
import torch
from typing import List, Tuple, Dict, Any, Optional
import threading

# 配置
MODEL_PATH = "model/SAILER_zh"

# 民事、刑事都分开路径
MINSHI_CORPUS_PATH = "/law/minshi.jsonl"
MINSHI_INDEX_PATH = "/law/minshi.faiss"
MINSHI_META_PATH = "/law/minshi.meta.jsonl"

XINGSHI_CORPUS_PATH = "/law/xingshi.jsonl"
XINGSHI_INDEX_PATH = "/law/xingshi.faiss"
XINGSHI_META_PATH = "/law/xingshi.meta.jsonl"

EMBED_DIM = 768

# 延迟加载
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModel] = None
_INDEX_CACHE = {}   # <--- 新增：分别缓存民事、刑事索引
_META_CACHE = {}
_INDEX_LOCK = threading.Lock()


def _ensure_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModel.from_pretrained(MODEL_PATH)
        _model.eval()
    return _tokenizer, _model


def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def encode_texts(texts: List[str], max_length: int = 512) -> np.ndarray:
    tokenizer, model = _ensure_model()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
    return _l2_normalize(emb, axis=1)


def _iter_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build_and_save_index(corpus_path: str, index_path: str, meta_path: str,
                         batch_size: int = 64) -> Tuple[faiss.IndexFlatIP, int]:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    index = faiss.IndexFlatIP(EMBED_DIM)
    total = 0

    with open(meta_path, 'w', encoding='utf-8') as fout:
        batch_texts, batch_meta = [], []

        for doc in _iter_jsonl(corpus_path):
            case_name = doc.get("case_name", "")
            content = doc.get("content", "")
            text = f"{case_name}\n{content}".strip()

            if not text:
                continue

            batch_texts.append(text)
            batch_meta.append({"case_name": case_name, "content": content})

            if len(batch_texts) >= batch_size:
                vecs = encode_texts(batch_texts)
                index.add(vecs)
                for m in batch_meta:
                    fout.write(json.dumps(m, ensure_ascii=False) + "\n")
                total += len(batch_texts)
                batch_texts, batch_meta = [], []

        if batch_texts:
            vecs = encode_texts(batch_texts)
            index.add(vecs)
            for m in batch_meta:
                fout.write(json.dumps(m, ensure_ascii=False) + "\n")
            total += len(batch_texts)

    faiss.write_index(index, index_path)
    print(f"索引已保存：{index_path}，文档数：{total}")
    print(f"元数据已保存：{meta_path}")

    return index, total


def load_index_and_meta(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    meta = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                meta.append(json.loads(line))
            except:
                continue
    return index, meta


def _ensure_index(index_path: str, meta_path: str):
    """为民事/刑事分别构建或加载索引 + 缓存"""
    key = index_path  # 用 index_path 做缓存 key

    if key in _INDEX_CACHE and key in _META_CACHE:
        return _INDEX_CACHE[key], _META_CACHE[key]

    with _INDEX_LOCK:
        # 双重检查
        if key in _INDEX_CACHE and key in _META_CACHE:
            return _INDEX_CACHE[key], _META_CACHE[key]

        # 若文件不存在，则自动构建
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            print(f"未找到索引：{index_path}，正在自动构建……")
            # 根据 index_path 选择对应的语料路径（更稳健的做法是把 corpus_path 传进来）
            corpus_path = MINSHI_CORPUS_PATH if "minshi" in index_path else XINGSHI_CORPUS_PATH
            # build_and_save_index 返回 (index, total)，构建完成后我们需要重新 load meta
            index, _ = build_and_save_index(corpus_path, index_path, meta_path)
            # 构建后从磁盘加载 meta（确保 meta 被正确赋值）
            _, meta = load_index_and_meta(index_path, meta_path)
        else:
            index, meta = load_index_and_meta(index_path, meta_path)

        # 缓存
        _INDEX_CACHE[key] = index
        _META_CACHE[key] = meta

        return index, meta


def case_retrieval(case_type: str, query: str) -> str:
    """支持民事 & 刑事的类案检索"""
    if case_type == "民事案件":
        index, meta = _ensure_index(MINSHI_INDEX_PATH, MINSHI_META_PATH)
    elif case_type == "刑事案件":
        index, meta = _ensure_index(XINGSHI_INDEX_PATH, XINGSHI_META_PATH)
    else:
        return "检索的类型错误，正确类型为：民事案件 或 刑事案件。"

    if index.ntotal == 0:
        return "索引为空。"

    vec = encode_texts([query])
    k = min(3, index.ntotal)
    D, I = index.search(vec, k)

    lines = ["检索得到的三个类案分别是："]
    for rank, idx in enumerate(I[0], 1):
        item = meta[idx]
        lines.append(f"{rank}. {item['case_name']}\n{item['content']}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(case_retrieval("民事案件", "借款合同纠纷"))
    print(case_retrieval("刑事案件", "盗窃罪"))
