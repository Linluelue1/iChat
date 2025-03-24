import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 配置 OpenAI 客户端
client = OpenAI(
    api_key="your-api-key-here",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 公共资源加载
class ResourceManager:
    def __init__(self):
        # 加载公共的嵌入模型
        self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # 加载各个模块的 FAISS 索引和知识库
        self.index_reading = faiss.read_index("ReadingChat/readingFaiss.index")
        with open("ReadingChat/readingKB.json", "r", encoding="utf-8") as f:
            self.kb_reading = json.load(f)

        self.index_writing = faiss.read_index("WritingChat/writingFaiss.index")
        with open("WritingChat/writingKB.json", "r", encoding="utf-8") as f:
            self.kb_writing = json.load(f)

        self.index_speaking = faiss.read_index("SpeakingChat/speakingFaiss.index")
        with open("SpeakingChat/speakingKB.json", "r", encoding="utf-8") as f:
            self.kb_speaking = json.load(f)

        self.index_listening = faiss.read_index("ListeningChat/listeningFaiss.index")
        with open("ListeningChat/listeningKB.json", "r", encoding="utf-8") as f:
            self.kb_listening = json.load(f)["documents"]
            with open("ListeningChat/listeningMB.json", "r", encoding="utf-8") as meta_f:
                meta_data = json.load(meta_f)
                self.file_map_listening = {entry["index_id"]: entry["knowledge_id"] for entry in meta_data["metadata"]}


resource_manager = ResourceManager()


# 阅读模块
@app.route('/api/reading', methods=['POST'])
def reading_api():
    data = request.json
    query = data.get('question', '')

    # 检索相关文档
    query_embedding = resource_manager.embed_model.encode([query]).astype("float32")
    indices, distances = resource_manager.index_reading.search(query_embedding, 3)
    retrieved_docs = []
    for idx in indices[0]:
        if idx < len(resource_manager.kb_reading):
            retrieved_docs.append(resource_manager.kb_reading[idx]["content"])
    context_str = "\n\n".join(retrieved_docs)

    # 构造提示：要求模型用结构化格式输出回答
    prompt = (
        "你是一个雅思阅读助手，请尽量用英文回答问题，并使用以下格式输出回答：\n\n"
        "**Question:** " + query + "\n\n"
        "**Answer:**\n"
        "（请在此处给出你的回答，分段落、条目或其他结构化形式展示）\n\n"
        "如果需要原文，请给出所有文章。\n\n"
        "请基于以下文档内容回答问题：\n" + context_str
    )

    # 调用模型
    completion = client.chat.completions.create(
        model="deepseek-r1",
        messages=[{"role": "user", "content": prompt}]
    )
    return jsonify({
        "status": "success",
        "answer": completion.choices[0].message.content
    })


# 写作模块
@app.route('/api/writing', methods=['POST'])
def writing_api():
    data = request.json
    query = data.get('question', '')

    # 检索相关文档
    query_vector = resource_manager.embed_model.encode([query])
    distances, indices = resource_manager.index_writing.search(query_vector, 3)
    relevant_docs = []
    for i in indices[0]:
        if i < len(resource_manager.kb_writing):
            relevant_docs.append(resource_manager.kb_writing[i])

    # 生成回答时拼接文档内容
    context_str = "\n".join([f"**Question:** {doc['Question']}\n**Essay:** {doc['Essay']}" for doc in relevant_docs])
    response = client.chat.completions.create(
        model="deepseek-r1",
        messages=[
            {"role": "system",
             "content": (
                 "你是一个雅思作文助手，请尽量用英文回答问题，并使用以下格式输出回答：\n\n"
                 "**Question:** 用户提问\n\n"
                 "**Essay:**\n"
                 "（请在此处给出你的作文回答，分段展示主要观点和论证）\n\n"
                 "**Key Features:**\n"
                 "1. 清晰的结构\n"
                 "2. 逻辑严谨\n"
                 "3. 精准的数据引用\n\n"
                 "如果需要原文，请给出所有文章。\n"
                 "请不要提供额外的解释。"
             )},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ],
        stream=False
    )
    return jsonify({
        "status": "success",
        "answer": response.choices[0].message.content
    })


# 口语模块
@app.route('/api/speaking', methods=['POST'])
def speaking_api():
    data = request.json
    query = data.get('question', '')

    # 检索相关文档
    query_vec = resource_manager.embed_model.encode([query], convert_to_numpy=True)
    distances, indices = resource_manager.index_speaking.search(query_vec, 3)
    retrieved_texts = []
    for idx in indices[0]:
        if idx < len(resource_manager.kb_speaking):
            retrieved_texts.append(resource_manager.kb_speaking[idx]["full_text"])
    context = "\n\n".join(retrieved_texts)

    # 构造对话历史：要求结构化输出
    conversation_history = [
        {"role": "system", "content": (
            "你是一个雅思口语助手，请尽量用英文回答问题，并使用以下格式输出回答：\n\n"
            "**Question:** 用户提问\n\n"
            "**Answer:**\n"
            "（请在此处给出你的口语回答，分段展示主要观点和说明）\n\n"
            "如果需要原文，请给出所有文章。\n"
            "请不要提供额外的解释。"
        )},
        {"role": "user", "content": f"背景知识：\n{context}\n\nQuestion: {query}"}
    ]

    # 调用模型
    completion = client.chat.completions.create(
        model="deepseek-r1",
        messages=conversation_history
    )
    return jsonify({
        "status": "success",
        "answer": completion.choices[0].message.content
    })


# 听力模块
@app.route('/api/listening', methods=['POST'])
def listening_api():
    data = request.json
    query = data.get('question', '')

    # 检索相关知识
    query_vec = resource_manager.embed_model.encode(query).reshape(1, -1)
    distances, indices = resource_manager.index_listening.search(query_vec, 3)
    results = []
    for idx in indices[0]:
        if idx in resource_manager.file_map_listening:
            doc = resource_manager.kb_listening[resource_manager.file_map_listening[idx]]
            results.append({
                "content": doc["content"],
                "metadata": {
                    "version": doc["cambridge_ver"],
                    "test": doc["test_num"],
                    "task": doc["task_num"],
                    "type": doc["content_type"]
                }
            })

    # 构建提示：要求模型输出结构化回答（注意：此处要求用中文回答）
    context = "\n\n".join([f"**[知识片段 {i + 1}]**:\n{item['content'][:500]}" for i, item in enumerate(results)])
    prompt = (
        "你是一个雅思听力助手，请基于以下文档内容回答问题，并用中文回答，回答需结构化显示（例如使用换行、加粗标题等）：\n\n"
        "**Question:** " + query + "\n\n"
        "**Answer:**\n"
        "（请在此处给出你的回答，分段展示主要观点和细节）\n\n"
        "如果需要原文，请给出所有文章。\n\n"
        "文档内容：\n" + context
    )

    # 调用模型
    completion = client.chat.completions.create(
        model="deepseek-r1",
        messages=[{"role": "user", "content": prompt}]
    )
    return jsonify({
        "status": "success",
        "answer": completion.choices[0].message.content
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
