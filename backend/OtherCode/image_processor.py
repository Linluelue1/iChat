import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import pandas as pd


class FaissIndexBuilder:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        # 加载多语言文本向量化模型（支持中文）
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def chunk_text(self, text, chunk_size=300, overlap=50):
        """智能文本分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # 尝试在句末分块
            if end < len(text):
                while end > start and text[end] not in {'.', '。', '\n', '！', '？'}:
                    end -= 1
                if end == start:  # 没有找到合适分隔符
                    end = start + chunk_size
            chunks.append(text[start:end].strip())
            start = end - overlap  # 设置重叠区间
        return chunks

    def build_index(self, text_content):
        """构建Faiss索引"""
        # 文本预处理和分块
        clean_text = text_content.replace('\n', ' ').replace('\r', '')
        chunks = self.chunk_text(clean_text)

        # 生成嵌入向量
        embeddings = self.model.encode(chunks,
                                       convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=True)

        # 转换为Faiss需要的格式
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        self.index.add(embeddings.astype(np.float32))

        # 保存元数据
        self.metadata = [{
            "text": chunk,
            "position": i
        } for i, chunk in enumerate(chunks)]

    def save_index(self, index_path="faiss_index"):
        """保存索引和元数据"""
        faiss.write_index(self.index, f"{index_path}.index")
        pd.DataFrame(self.metadata).to_csv(f"{index_path}_metadata.csv", index=False)

    def load_index(self, index_path="faiss_index"):
        """加载索引"""
        self.index = faiss.read_index(f"{index_path}.index")
        self.metadata = pd.read_csv(f"{index_path}_metadata.csv").to_dict('records')


# 使用示例
if __name__ == "__main__":
    # 读取OCR提取的文本
    with open("extracted_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 构建索引
    index_builder = FaissIndexBuilder()
    index_builder.build_index(text)
    index_builder.save_index()

    # 检索示例（可单独使用）
    query = "T4签证相关要求"
    query_vector = index_builder.model.encode([query])
    query_vector = normalize(query_vector, norm='l2')  # 归一化

    D, I = index_builder.index.search(query_vector.astype(np.float32), k=3)

    print("\nTop 3 相关结果：")
    for idx in I[0]:
        print(f"[相似度：{D[0][idx]:.4f}] {index_builder.metadata[idx]['text'][:100]}...")