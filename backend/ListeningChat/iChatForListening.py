import json
import faiss
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class IELTSAssistant:
    def __init__(self, api_key):
        # 初始化模型组件
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.index = faiss.read_index("listeningFaiss.index")

        # 加载元数据
        with open("listeningMB.json", "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            self.metadata = meta_data["metadata"]
            self.file_map = {entry["index_id"]: entry["knowledge_id"]
                           for entry in self.metadata}

        # 加载知识库
        with open("listeningKB.json", "r", encoding="utf-8") as f:
            self.knowledge = json.load(f)["documents"]

        # 初始化DeepSeek客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 对话历史
        self.history = [
            {"role": "system", "content": "你是一位专业的雅思考试助手，根据知识库内容回答问题"}
        ]

    def _retrieve_knowledge(self, query, k=3):
        """检索相关知识"""
        query_vec = self.model.encode(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, k)
        results = []
        for idx in indices[0]:
            if idx in self.file_map:
                doc = self.knowledge[self.file_map[idx]]
                results.append({
                    "content": doc["content"],
                    "metadata": {
                        "version": doc["cambridge_ver"],
                        "test": doc["test_num"],
                        "task": doc["task_num"],
                        "type": doc["content_type"]
                    }
                })
        return results
    def _format_prompt(self, query, knowledge):
        """构建带知识上下文的提示"""
        context = "\n\n".join([f"[知识片段 {i + 1}]:\n{item['content'][:500]}"
                               for i, item in enumerate(knowledge)])
        return f"""请根据以下雅思知识库内容回答问题：
        {context}
        用户问题：{query}
        请用中文回答，保持专业且易懂："""

    def ask(self, query):
        """执行完整问答流程"""
        # 知识检索（保持不变）
        knowledge = self._retrieve_knowledge(query)

        # 确保提示内容为字符串
        current_prompt = str(self._format_prompt(query, knowledge))

        # 维护对话历史时显式转换为字符串
        self.history.append({"role": "user", "content": str(current_prompt)})
        self.history = self.history[-6:]

        try:
            # 调用前验证消息格式
            for msg in self.history:
                if not isinstance(msg["content"], str):
                    msg["content"] = str(msg["content"])

            completion = self.client.chat.completions.create(
                model="deepseek-r1",
                messages=self.history,
                temperature=0.7,
                max_tokens=1000
            )
            answer = completion.choices[0].message.content

            # 更新对话历史
            self.history.append({"role": "assistant", "content": answer})
            return answer

        except Exception as e:
            return f"请求出错：{str(e)}"




def main():
    # 初始化配置
    api_key = "sk-d8aa43d322ba44b3b105b98feeb142a6"  # 替换为实际API密钥
    assistant = IELTSAssistant(api_key)

    print("雅思考试助手已启动（输入exit退出）")
    print("=" * 40)

    # 对话循环
    while True:
        try:
            query = input("\n你的问题：")
            if query.lower() in ["exit", "quit"]:
                break

            if not query.strip():
                continue

            response = assistant.ask(query)
            print("\n助手回答：")
            print(response)
            print("=" * 40)

        except KeyboardInterrupt:
            print("\n对话已终止")
            break


if __name__ == "__main__":
    main()