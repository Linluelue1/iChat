import logging
from praisonaiagents import Agent
import requests
# 定义配置
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "praison",
            "path": ".praison"
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "deepseek-r1:latest",  # 确保模型名称正确
            "temperature": 0,
            "max_tokens": 8000,
            "ollama_base_url": "http://localhost:11434",  # 确保 URL 正确
        }
    },

    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
            "embedding_dims": 1536
        }
    }
}
logging.basicConfig(level=logging.DEBUG)
# 初始化智能体
agent = Agent(
    name="知识智能体",
    instructions="回答问题。",
    #knowledge=["dataSet/knowledge_base1.json"],  # 替换为您的 PDF
    knowledge_config=config,
    user_id="user1",
    llm="deepseek-r1"
)

# 打印配置信息（调试用）
print("Ollama 配置：", config["llm"]["config"])

# 提问
response = agent.start("我对于第一题有点不了解 你能解释一下吗")
if response is not None:
    print(response)
else:
    print("请求失败，响应为 None")