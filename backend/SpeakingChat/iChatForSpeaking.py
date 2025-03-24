import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/SpeakingChat', methods=['POST'])
def api_chat():
    try:
        data = request.json
        answer = chat_with_model(data['question'])
        return jsonify({
            "status": "success",
            "answer": answer  # 确保所有模块使用相同的字段名
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ---------------------------
# 1. 加载知识库和FAISS索引
# ---------------------------
with open("SpeakingChat/speakingKB.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# 加载之前保存的FAISS索引，注意索引文件格式为 .index
index = faiss.read_index("SpeakingChat/speakingFaiss.index")

# ---------------------------
# 2. 加载嵌入模型，用于生成查询向量
# ---------------------------
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ---------------------------
# 3. 初始化 OpenAI 客户端（兼容阿里云Dashscope服务）
# ---------------------------
client = OpenAI(
    api_key="sk-d8aa43d322ba44b3b105b98feeb142a6",  # 或直接写入你的 API Key，如 api_key="sk-xxx"
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ---------------------------
# 4. 初始化多轮对话的对话历史（system消息可用于指导模型行为）
# ---------------------------
conversation_history = [
    {"role": "system", "content": "你是一个雅思口语学习助手，请结合下方背景知识回答用户问题。"}
]


# ---------------------------
# 5. 定义一个函数，根据用户输入检索相关知识
# ---------------------------
def retrieve_context(query, top_k=3):
    # 生成用户查询的向量
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    retrieved_texts = []
    # 注意：检索到的索引与知识库条目一一对应
    for idx in indices[0]:
        if idx < len(knowledge_base):
            retrieved_texts.append(knowledge_base[idx]["full_text"])
    # 将检索结果合并为一个背景文本
    context = "\n\n".join(retrieved_texts)
    return context


# ---------------------------
# 6. 定义调用模型进行多轮对话的函数
# ---------------------------
def chat_with_model(user_query):
    # 检索相关背景知识
    context = retrieve_context(user_query, top_k=3)

    # 构造本次对话内容，将背景知识和用户问题合并
    # 这里将背景知识放在问题前，帮助模型结合上下文回答
    user_message = f"背景知识：\n{context}\n\n问题：{user_query}"

    # 将用户消息加入多轮对话历史
    conversation_history.append({"role": "user", "content": user_message})

    # 调用模型，传入对话历史（多轮对话支持历史累积）
    completion = client.chat.completions.create(
        model="deepseek-r1",  # 可按需更换模型名称
        messages=conversation_history
    )

    # 提取思考过程和最终答案
    reasoning = completion.choices[0].message.reasoning_content
    answer = completion.choices[0].message.content

    # 将模型回复也加入对话历史，方便后续多轮对话
    conversation_history.append({"role": "assistant", "content": answer})

    # 打印思考过程和答案
    # print("思考过程：")
    # print(reasoning)
    print("\n最终答案：")
    print(answer)

    return answer


# ---------------------------
# 7. 交互式对话循环（可多轮咨询）
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5003)
    args = parser.parse_args()
    app.run(port=args.port, debug=True)  # 启用调试模式
