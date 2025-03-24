import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json


def parse_file(filename):
    """
    解析雅思口语题目文本文件：
      - 遇到形如 “数字 P1 标题” 的行，则为 Part1 题目开始
      - 遇到形如 “数字 P2 标题” 的行，则为 Part2&3 题目开始
      - 在 Part2&3 中，当遇到单独一行 "P3" 时，表示从此开始进入 Part3 部分
      - 其他内容则归属到当前题目的对应部分中
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    entries = []
    current_entry = None
    current_mode = None  # 用于区分当前处于 p1 / p2 / p3 内容
    id_counter = 1

    # 匹配题目头部的正则，格式例如："1 P1 Colours" 或 "1 P2 精力充沛的人"
    header_pattern = re.compile(r"^(\d+)\s+(P1|P2)\s+(.*)$")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 如果行以 “一、”, “二、”, “三、”, “四、” 等开头，则视为分区标识，跳过
        if re.match(r"^[一二三四、]", line):
            continue

        # 判断是否为单独的 "P3" 行（Part2&3 中的分界标识）
        if line == "P3":
            if current_entry is not None and current_entry.get("part") == "Part2&3":
                current_mode = "p3"
            continue

        # 判断是否为题目头部
        m = header_pattern.match(line)
        if m:
            # 如果已有上一个题目，则将其保存
            if current_entry is not None:
                entries.append(current_entry)
            num, p_tag, title = m.groups()
            if p_tag == "P1":
                current_entry = {
                    "id": id_counter,
                    "part": "Part1",
                    "title": title,
                    "content": []  # 用于保存该题的所有问题文本
                }
                current_mode = "p1"
            elif p_tag == "P2":
                current_entry = {
                    "id": id_counter,
                    "part": "Part2&3",
                    "title": title,
                    "p2": [],  # Part2 部分
                    "p3": []  # Part3 部分
                }
                current_mode = "p2"
            id_counter += 1
            continue

        # 非标题行，则归入当前题目中
        if current_entry is not None:
            if current_entry["part"] == "Part1":
                current_entry["content"].append(line)
            elif current_entry["part"] == "Part2&3":
                if current_mode == "p2":
                    current_entry["p2"].append(line)
                elif current_mode == "p3":
                    current_entry["p3"].append(line)
                else:
                    # 默认归入 p2
                    current_entry["p2"].append(line)
    # 将最后一个题目加入
    if current_entry is not None:
        entries.append(current_entry)
    return entries


# 1. 解析文件，生成条目列表
entries = parse_file("ielts_speaking_questions1.txt")

# 2. 构造知识库，每个条目增加一个 full_text 字段，作为后续检索的全文内容
knowledge_base = []
documents = []  # 用于存储每个条目的全文，用于生成嵌入向量

for entry in entries:
    if entry["part"] == "Part1":
        full_text = entry["title"] + "\n" + "\n".join(entry["content"])
    elif entry["part"] == "Part2&3":
        full_text = entry["title"] + "\n" + "P2:\n" + "\n".join(entry["p2"]) + "\n" + "P3:\n" + "\n".join(entry["p3"])
    # 将合并的全文存入当前条目中
    entry["full_text"] = full_text
    knowledge_base.append(entry)
    documents.append(full_text)

# 3. 保存知识库到 JSON 文件中
with open("speakingKB.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

print("知识库已保存到 readingKB.json")

# 4. 利用 SentenceTransformer 为每个条目的全文生成嵌入向量
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(documents, convert_to_numpy=True)

# 5. 构建 FAISS 索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 6. 将 FAISS 索引保存为 .index 文件
faiss.write_index(index, "speakingFaiss.index")
print("FAISS 索引已保存到 speakingFaiss.index")
