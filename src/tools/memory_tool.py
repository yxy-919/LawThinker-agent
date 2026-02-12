# 在不断探索和核查的过程中遇到关键信息则进行存储
# 知识存储：关键的法条、案例知识（每个agent有一个knowledge base）
# 上下文存储：存储收集得到的信息（每个agent有一个context base）
def storeMemory(memory_type, content, memory_block):
    # 确保 key 存在
    if "知识存储" not in memory_block:
        memory_block["知识存储"] = []
    if "上下文存储" not in memory_block:
        memory_block["上下文存储"] = []

    if memory_type == "知识存储":
        memory_block["知识存储"].append(content)
        return f"知识存储成功：{content}"
    elif memory_type == "上下文存储":
        memory_block["上下文存储"].append(content)
        return f"上下文存储成功：{content}"
    else:
        return f"没有这种存储类型：{memory_type}。"

def fetchMemory(memory_type,memory_block):
    if memory_type == "知识存储":
        return "知识存储：" + "\n".join(memory_block["知识存储"])
    elif memory_type == "上下文存储":
        return "上下文存储：" + "\n".join(memory_block["上下文存储"])
    else:
        return "没有这种存储类型。"