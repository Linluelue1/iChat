import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class IELTSKnowledgeBase:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.index = faiss.IndexFlatL2(384)  # 匹配模型维度
        self.metadata = []
        self.knowledge = []

    def process_directory(self, base_dir="dataset"):
        for root, _, files in os.walk(base_dir):
            for filename in tqdm(files, desc="Processing files"):
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    self._process_file(file_path)

    def _parse_metadata(self, file_path):
        """增强版元数据解析"""
        path_parts = file_path.split(os.sep)
        metadata = {
            "cambridge_ver": None,
            "test_num": None,
            "task_num": None,
            "content_type": None,
            "full_path": file_path
        }

        # 解析路径结构（示例路径：dataset/剑桥雅思4/Test1/Task1/5-4-1-1听力原文.txt）
        for part in path_parts:
            if "剑桥雅思" in part:
                metadata["cambridge_ver"] = part.split("剑桥雅思")[-1]
            elif part.startswith("Test"):
                metadata["test_num"] = part[4:]
            elif part.startswith("Task"):
                metadata["task_num"] = part[4:]

        # 解析文件名内容类型
        if "听力原文" in file_path:
            metadata["content_type"] = "transcript"
        elif "听力答案" in file_path:
            metadata["content_type"] = "answer"

        return metadata

    def _process_file(self, file_path):
        """处理单个文件并建立映射关系"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 生成元数据
        metadata = self._parse_metadata(file_path)
        metadata["content"] = content[:10000]  # 限制内容长度

        # 存储知识条目
        knowledge_id = len(self.knowledge)
        self.knowledge.append(metadata)

        # 生成向量并建立索引映射
        embedding = self.model.encode(content)
        embedding = embedding.reshape(1, -1).astype('float32')

        # 添加索引并记录映射关系
        if self.index.ntotal == 0:
            self.index.add(embedding)
        else:
            self.index.add(embedding)

        # 维护索引ID到知识库ID的映射
        self.metadata.append({
            "index_id": self.index.ntotal - 1,
            "knowledge_id": knowledge_id
        })

    def save(self):
        """保存完整知识体系"""
        # 保存FAISS索引（使用.index后缀）
        faiss.write_index(self.index, "listeningFaiss.index")

        # 保存元数据映射
        with open("listeningMB.json", "w", encoding="utf-8") as f:
            json.dump({
                "metadata": self.metadata  # 确保数据结构为字典
            }, f, ensure_ascii=False)

        # 保存完整知识库
        with open("listeningKB.json", "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.knowledge,
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "dimension": 384
            }, f, ensure_ascii=False, indent=2)


# 使用示例
if __name__ == "__main__":
    kb = IELTSKnowledgeBase()
    kb.process_directory()
    kb.save()
    print(f"构建完成：知识库条目={len(kb.knowledge)}, 索引数量={kb.index.ntotal}")