import pandas as pd
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 加载 CSV 文件
df = pd.read_csv("../dataSet/ielts_writing_dataset.csv", na_values=["", " ", "NaN"])

# 删除 Essay 或 Question 列为空的行
df.dropna(subset=["Essay", "Question"], inplace=True)

# 提取 Question 和 Essay 列
knowledge_base = df[["Question", "Essay"]].to_dict("records")

# 保存知识库到 JSON 文件
with open("writingKB.json", "w") as f:
    json.dump(knowledge_base, f)

print("知识库已保存为 knowledge_base1.json")

# 加载预训练的句子编码模型
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# 将知识库内容编码为向量
documents = [f"Question: {item['Question']}\nEssay: {item['Essay']}" for item in knowledge_base]
document_vectors = encoder.encode(documents)

# 构建 FAISS 索引
dimension = document_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离
index.add(document_vectors)

# 保存 FAISS 索引到文件
faiss.write_index(index, "speakingFaiss.index")

print("FAISS 索引已保存为 speakingFaiss.index")