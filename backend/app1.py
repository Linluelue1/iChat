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
    api_key="sk-d8aa43d322ba44b3b105b98feeb142a6",
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
        with open("WritingChat/writingKB.json", "r") as f:
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

    # 构造提示
    prompt = (f"你是一个雅思阅读助手 请你尽量用英文回答问题"
              f"如果用户需要原文请给出所有的文章"
              f"没有强调给出解释或答案或相同字眼时请不要给出解释 除非用户让你这样做"
              f"请基于以下文档内容回答问题：\n\n{context_str}\n\n问题：{query}")

    # 调用模型
    completion = client.chat.completions.create(
        model="deepseek-r1",
        messages=[
            {"role": "user", "content": prompt}
        ]
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

    # 生成回答
    context_str = "\n".join([f"Question: {doc['Question']}\nEssay: {doc['Essay']}" for doc in relevant_docs])
    response = client.chat.completions.create(
        model="deepseek-r1",
        messages=[
            {"role": "system",
             "content": f"你是一个雅思作文助手 请你尽量用英文回答问题"
              f"如果用户需要原文请给出所有的文章"
              f"没有强调给出解释或答案或相同字眼时请不要给出解释 除非用户让你这样做"},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
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

    # 构造对话历史
    conversation_history = [
        {"role": "system", "content": f"你是一个雅思口语助手 请你尽量用英文回答问题"
              f"如果用户需要原文请给出所有的文章"
              f"没有强调给出解释或答案或相同字眼时请不要给出解释 除非用户让你这样做"},
        {"role": "user", "content": f"背景知识：\n{context}\n\n问题：{query}"},
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

    # 构建提示
    context = "\n\n".join([f"[知识片段 {i + 1}]:\n{item['content'][:500]}" for i, item in enumerate(results)])
    prompt = (f"你是一个雅思阅读助手 请你尽量用英文回答问题"
              f"如果用户需要原文请给出所有的文章"
              f"没有强调给出解释或答案或相同字眼时请不要给出解释 除非用户让你这样做"
              f"请基于以下文档内容回答问题：\n{context}\n用户问题：{query}\n请用中文回答，保持专业且易懂：")

    # 调用模型
    completion = client.chat.completions.create(
        model="deepseek-r1",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return jsonify({
        "status": "success",
        "answer": completion.choices[0].message.content
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)