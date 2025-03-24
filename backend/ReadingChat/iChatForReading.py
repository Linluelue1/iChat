import os
import json
import faiss
import numpy as np
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS  # 添加在文件开头

# 初始化Flask应用（添加在现有代码最上方）
app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 初始化 OpenAI 客户端
# 注意：这里的 base_url 链接可能存在问题，导致无法正常解析。
# 如果遇到问题，请检查链接的合法性或稍后重试。
client = OpenAI(
    api_key="sk-d8aa43d322ba44b3b105b98feeb142a6",  # 使用环境变量获取 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 设置你的基础 API URL
)


def load_knowledge_base(kb_file="knowledge_base1.json"):
    """
    加载知识库 JSON 文件，返回其中的文档列表。

    参数:
        kb_file (str): 知识库文件的路径，默认为 "knowledge_base1.json"。

    返回:
        list: 包含文档信息的列表，每个文档是一个字典，包含 "file_path" 和 "content"。
    """
    with open(kb_file, "r", encoding="utf-8") as f:
        kb = json.load(f)
    return kb


def load_faiss_index(index_file="readingFaiss.index"):
    """
    加载 FAISS 索引文件，用于后续的相似性搜索。

    参数:
        index_file (str): FAISS 索引文件的路径，默认为 "readingFaiss.index"。

    返回:
        faiss.Index: 加载的 FAISS 索引对象。
    """
    index = faiss.read_index(index_file)
    return index


def search_index(query_embedding, index, top_k=5):
    """
    在 FAISS 索引中检索与查询嵌入最相似的文档。

    参数:
        query_embedding (np.ndarray): 查询文本的嵌入向量。
        index (faiss.Index): 加载的 FAISS 索引对象。
        top_k (int): 返回的最相似文档数量，默认为 5。

    返回:
        tuple: 包含两个值：
            - indices (np.ndarray): 最相似文档的索引。
            - distances (np.ndarray): 与查询嵌入的距离（相似度）。
    """
    distances, indices = index.search(query_embedding, top_k)
    return indices, distances


def summarize_text(text, max_length=200):
    """
    简单截断文本到指定的最大长度，用于简化输出。

    参数:
        text (str): 需要截断的文本。
        max_length (int): 截断的最大长度，默认为 200。

    返回:
        str: 截断后的文本。
    """
    return text if len(text) <= max_length else text[:max_length] + "..."


def query_deepseek_with_context(query, index_file="readingFaiss.index", kb_file="knowledge_base1.json"):
    """
    根据用户输入的查询，结合知识库和 FAISS 索引，调用外部 API（如 DeepSeek）生成回答。

    参数:
        query (str): 用户输入的查询文本。
        index_file (str): FAISS 索引文件路径，默认为 "readingFaiss.index"。
        kb_file (str): 知识库文件路径，默认为 "knowledge_base1.json"。

    返回:
        str: 从外部 API 获得的最终回答。
    """
    # 1. 加载知识库和 FAISS 索引
    kb = load_knowledge_base(kb_file)
    index = load_faiss_index(index_file)

    # 2. 利用 SentenceTransformer 模型计算查询嵌入
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query]).astype("float32")
    query_embedding = np.array(query_embedding)

    # 3. 在 FAISS 索引中检索最相近的文档（取 top 3）
    indices, distances = search_index(query_embedding, index, top_k=3)

    # 4. 根据检索到的索引，从知识库中获取对应文档内容，并进行截断
    retrieved_docs = []
    for idx in indices[0]:
        if idx < len(kb):
            summarized = summarize_text(kb[idx]["content"], max_length=1000)
            retrieved_docs.append(summarized)

    # 拼接文档内容作为上下文信息
    context_str = "\n\n".join(retrieved_docs)

    # 构造最终的 prompt，将上下文和用户问题结合
    prompt = f"请基于以下文档内容回答问题：\n\n{context_str}\n\n问题：{query}"

    try:
        # 5. 调用 DeepSeek API（使用 deepseek-r1 模型，可根据实际情况调整）
        completion = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )

        # 使用 getattr 安全获取 reasoning_content（若无则返回“无思考过程”）
        reasoning = getattr(completion.choices[0].message, 'reasoning_content', "无思考过程")
        final_answer = completion.choices[0].message.content

        # print("思考过程：")
        # print(reasoning)
        # print("最终答案：")
        # print(final_answer)

        return final_answer
    except Exception as e:
        print("调用 OpenAI API 时出错：", e)
        return None


@app.route('/api/reading', methods=['POST'])
def handle_reading():
    data = request.json
    answer = query_deepseek_with_context(data['question'])
    return jsonify({"answer": answer})

if __name__ == "__main__":
    # 原来的命令行交互模式改为：
    app.run(host='0.0.0.0', port=5000, debug=True)