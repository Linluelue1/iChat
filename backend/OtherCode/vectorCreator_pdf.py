import json
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def extract_sections(pdf_path):
    """
    基于样式特征提取结构化内容
    返回格式：[{
        "section_type": "听力|阅读|写作|口语",
        "subsection": "篇章结构|场景背景|必背词汇...",
        "content": "具体内容"
    }]
    """
    hierarchical_pattern = re.compile(r'^(【(听力|阅读|写作|口语)】)|^▶\s+(篇章结构|场景背景|必背词汇|难句解析|试题分析)')

    with pdfplumber.open(pdf_path) as pdf:
        sections = []
        current_section = {}

        for page in pdf.pages:
            text = page.extract_text()
            for line in text.split('\n'):
                # 检测章节标题
                if match := re.match(r'^【(听力|阅读|写作|口语)】(.*)', line):
                    current_section = {
                        "section_type": match.group(1),
                        "subsections": []
                    }
                    sections.append(current_section)
                # 检测子模块标题
                elif match := re.match(r'▶\s+(.*?)\s+(篇章结构|场景背景|必背词汇|难句解析|试题分析)', line):
                    current_subsection = {
                        "subsection_type": match.group(2),
                        "content": []
                    }
                    current_section["subsections"].append(current_subsection)
                # 内容收集
                elif current_section.get("subsections"):
                    current_subsection["content"].append(line.strip())

    # 后处理：合并分段内容
    for section in sections:
        for sub in section["subsections"]:
            sub["content"] = "\n".join(sub["content"])
    return sections


def structure_features(raw_sections):
    """
    将原始解析内容转换为结构化特征
    返回格式：[{
        "section": "听力Section1",
        "question_type": "表格填空",
        "key_points": ["租房场景", "地址填写"],
        "keywords": ["homestay", "deposit"],
        "analysis": "详细解析内容...",
        "sample_answer": "参考答案..."
    }]
    """
    structured_data = []

    for section in raw_sections:
        # 提取公共特征
        section_type = section["section_type"]

        for sub in section["subsections"]:
            # 根据子模块类型处理
            if sub["subsection_type"] == "篇章结构":
                metadata = {
                    "question_type": re.search(r'题型：(.*?)\n', sub["content"]).group(1),
                    "key_points": re.findall(r'考查重点：(.+?)(?=\n|$)', sub["content"])
                }
            elif sub["subsection_type"] == "必背词汇":
                keywords = re.findall(r'\| (.+?) \|', sub["content"])
            elif sub["subsection_type"] == "试题分析":
                analysis = re.sub(r'^【解析】', '', sub["content"])

                # 构建完整条目
                structured_data.append({
                    "section": f"{section_type} {metadata['question_type']}",
                    **metadata,
                    "keywords": keywords,
                    "analysis": analysis,
                    "sample_answer": re.search(r'参考答案：(.+)', sub["content"]).group(1)
                })

    return structured_data


def build_knowledge_base(structured_data):
    try:
        # === 数据校验 ===
        if not structured_data:
            raise ValueError("结构化数据为空，请检查PDF解析结果")

        # 检查数据结构
        sample_item = structured_data[0]
        required_keys = ['section', 'key_points', 'analysis']
        for key in required_keys:
            if key not in sample_item:
                raise KeyError(f"数据条目缺少必要字段: {key}")

        # === 文档生成 ===
        documents = []
        for idx, item in enumerate(structured_data, 1):
            doc = f"{item['section']} {' '.join(item['key_points'])} {item['analysis']}"
            if len(doc.strip()) < 10:  # 检查有效内容
                print(f"警告：第 {idx} 条数据内容过短")
            documents.append(doc)
            if idx <= 3:  # 打印前3条样本
                print(f"[样本{idx}]: {doc[:80]}...")

        # === 向量编码 ===
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("\n开始编码文档...")
        vectors = encoder.encode(
            documents,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        # === 维度验证 ===
        print(f"\n编码完成，向量矩阵形状：{vectors.shape}")
        if len(vectors.shape) != 2:
            raise ValueError(f"非预期的向量维度：{vectors.shape}，应为二维矩阵")
        if vectors.shape[0] != len(documents):
            raise ValueError(f"向量数量({vectors.shape[0]})与文档数({len(documents)})不匹配")

        # === 索引构建 ===
        dimension = vectors.shape[1]
        print(f"向量维度：{dimension}")

        index = faiss.IndexFlatL2(dimension)
        index.add(vectors.astype(np.float32))

        # === 保存结果 ===
        faiss.write_index(index, "ielts_t4_index.faiss")
        with open("ielts_t4_knowledge.json", "w") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)

        print(f"\n成功构建知识库：{len(documents)} 条数据 | 索引维度 {dimension}D")

    except Exception as e:
        print(f"\n错误发生：{str(e)}")
        print("调试信息：")
        print(f"结构化数据类型：{type(structured_data)}")
        if structured_data:
            print(f"首条数据结构：{json.dumps(structured_data[0], indent=2, ensure_ascii=False)}")
        raise


if __name__ == "__main__":
    # Step 1: 解析PDF
    raw_data = extract_sections("../dataSet/T4精讲.pdf")

    # Step 2: 结构化处理
    structured_data = structure_features(raw_data)

    # Step 3: 构建知识库
    build_knowledge_base(structured_data)