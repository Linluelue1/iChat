import os
import json
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_documents(data_dir="data"):
    """
    遍历 data 目录下所有 txt 文件，并读取内容。
    返回一个列表，每个元素包含文件路径和内容。
    """
    documents = []
    # 递归遍历所有子目录中的 txt 文件
    for file_path in glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append({
            "file_path": file_path,
            "content": content
        })
    return documents


def build_faiss_index(documents, model_name="all-MiniLM-L6-v2"):
    """
    利用 SentenceTransformer 模型计算每个文档的嵌入，并构建 FAISS 索引。
    返回构建好的索引和所有嵌入矩阵。
    """
    # 加载预训练模型
    model = SentenceTransformer(model_name)
    texts = [doc["content"] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


def save_index(index, index_path="readingFaiss.index"):
    """
    将 FAISS 索引保存到磁盘
    """
    faiss.write_index(index, index_path)


def save_knowledge_base(documents, kb_path="knowledge_base1.json"):
    """
    将知识库（文档列表）以 JSON 格式保存
    """
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 指定数据目录（根据你的文件结构，如 data/section5/...）
    data_dir = "data"
    # 加载所有 txt 文档
    documents = load_documents(data_dir)
    print(f"共加载 {len(documents)} 个文档。")

    # 构建 FAISS 索引
    index, embeddings = build_faiss_index(documents)
    print("FAISS 索引构建完成。")

    # 保存索引和知识库
    save_index(index, "readingFaiss.index")
    save_knowledge_base(documents, "readingKB.json")
    print("FAISS 索引和知识库 JSON 文件已保存。")
