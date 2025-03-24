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
    knowledge=["dataSet/knowledge_base1.json"],  # 替换为您的 PDF
    knowledge_config=config,
    user_id="user1",
    llm="deepseek-r1"
)

import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "deepseek-r1",
    "prompt": "我对于第一题有点不太理解 你能解释一下吗"
}
response = requests.post(url, json=payload)
print(response.text)