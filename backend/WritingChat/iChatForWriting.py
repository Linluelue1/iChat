import faiss
import os
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 加载预训练的句子编码模型
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# 加载 FAISS 索引
index = faiss.read_index("speakingFaiss.index")

# 加载知识库
with open("writingKB.json", "r") as f:
    knowledge_base = json.load(f)

# 初始化 OpenAI 客户端
client = OpenAI(api_key="sk-d8aa43d322ba44b3b105b98feeb142a6", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def retrieve_documents(query, top_k=3):
    """
    根据用户输入检索相关文档。
    """
    # 将用户输入编码为向量
    query_vector = encoder.encode([query])

    # 使用 FAISS 检索最相关的文档
    distances, indices = index.search(query_vector, top_k)

    # 返回相关文档
    results = []
    for i in indices[0]:
        if i < len(knowledge_base):  # 关键检查！
            results.append(knowledge_base[i])
    return results

def generate_response(query, context):
    """
    根据检索到的文档生成回答。
    """
    # 将检索到的文档作为上下文
    context_str = "\n".join([f"Question: {doc['Question']}\nEssay: {doc['Essay']}" for doc in context])

    # 调用生成模型
    response = client.chat.completions.create(
        model="deepseek-r1",
        messages=[
            {"role": "system", "content": "You are a IELTS test1 assistant. "
                                          "Do not answer any things which does not matter with ielts."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
        ],
        stream=False
    )
    return response


def chat_with_assistant():
    """
    交互式对话程序。
    """
    print("Welcome to the iChat! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # 检索相关文档
        relevant_docs = retrieve_documents(user_input)

        # 生成回答
        response = generate_response(user_input, relevant_docs)
        # 通过reasoning_content字段打印思考过程
        print("思考过程：")
        print(response.choices[0].message.reasoning_content)
        print(f"Assistant: {response.choices[0].message.content}")


if __name__ == "__main__":
    chat_with_assistant()