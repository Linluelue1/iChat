import streamlit as st
from praisonaiagents import Agent
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import threading


def worker():
    ctx = get_script_run_ctx()
    # 在线程中执行需要的逻辑
    add_script_run_ctx(threading.current_thread(), ctx)


def init_agent():
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
                "model": "deepseek-r1:latest",
                "temperature": 0,
                "max_tokens": 8000,
                "ollama_base_url": "http://localhost:11434",
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
    return Agent(
        name="雅思助手",
        instructions="根据提供的雅思知识库回答问题。",
        knowledge=["G:\\GProgramme\\graduatePaper\\pythonProject\\ReadingChat\\data\\section5\\test1\\passage1.txt",
                   "G:\\GProgramme\\graduatePaper\\pythonProject\\ReadingChat\\data\\section5\\test1\\passage2.txt"],
        knowledge_config=config,
        user_id="user1",
        llm="deepseek-r1"
    )


thread = threading.Thread(target=worker)
thread.start()
thread.join()

st.title("雅思学习助手")
agent = init_agent()

prompt = st.text_input("请输入您的问题：")
if prompt:
    response = agent.start(prompt)
    st.write("AI 回答：", response)
