import re
import json
import numpy as np
import faiss


def parse_txt_to_kb(filename):
    """
    解析 txt 文档，将题目整理为知识库 JSON 格式：
    - part1：包括文档中所有标记为 “P1” 或以 “万年老题” 开头的题目。
      每个主题格式为 { "topic": 主题, "part1 questions": [问题列表] }
    - part2&part3：将每个主题下 P2 和 P3 的问题分别归类，
      格式为 { "topic": 主题, "part2 questions": [P2 问题列表], "part3 questions": [P3 问题列表] }
    """
    kb = {"part1": [], "part2&part3": []}
    current_section = None  # "part1" 或 "part2&part3"
    current_topic = None  # 当前主题字典
    current_state = None  # 对于 part2&part3，标识当前录入的是 "part2" 还是 "part3"

    # 匹配章节标题，例如 “一、 Part 1 …” 或 “四、 Part2&Part3 …”
    section_header_pattern = re.compile(r'^[一二三四五六七八九十]+\s*[、.]\s*Part')
    # 匹配 Part1 主题行，例如 “1 P1 Colours”
    part1_topic_pattern = re.compile(r'^\d+\s*P1\s+(.*)')
    # 匹配 Part2 主题行，例如 “1 P2 学到新东西的网络视频”
    part2_topic_pattern = re.compile(r'^\d+\s*P2\s+(.*)')

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 如果是章节标题（例如 “四、 Part2&Part3 老题沿用（76 道）”），则根据内容切换当前 section，并跳过该行
        if section_header_pattern.match(line):
            if "Part 1" in line or "Part1" in line:
                current_section = "part1"
            elif "Part 2&3" in line or "Part2&Part3" in line:
                current_section = "part2&part3"
            current_topic = None  # 切换章节时清空当前主题
            current_state = None
            continue

        # 对于 Part1 部分
        if current_section == "part1":
            m = part1_topic_pattern.match(line)
            if m:
                # 新主题开始，保存上一个主题
                if current_topic:
                    kb["part1"].append(current_topic)
                topic_title = m.group(1).strip()
                current_topic = {"topic": topic_title, "part1 questions": []}
            else:
                if current_topic is not None:
                    current_topic["part1 questions"].append(line)
            continue

        # 对于 Part2&Part3 部分
        if current_section == "part2&part3":
            m = part2_topic_pattern.match(line)
            if m:
                # 新主题开始，保存前一个主题
                if current_topic:
                    kb["part2&part3"].append(current_topic)
                topic_title = m.group(1).strip()
                current_topic = {"topic": topic_title, "part2 questions": [], "part3 questions": []}
                current_state = "part2"  # 默认开始录入 P2 问题
                continue
            # 如果行正好为 "P3"，则切换录入状态到 part3
            if line == "P3":
                current_state = "part3"
                continue
            # 如果行同时匹配 P2 主题（可能重复出现），也作为新主题处理
            m2 = part2_topic_pattern.match(line)
            if m2:
                if current_topic:
                    kb["part2&part3"].append(current_topic)
                topic_title = m2.group(1).strip()
                current_topic = {"topic": topic_title, "part2 questions": [], "part3 questions": []}
                current_state = "part2"
                continue
            # 否则视为当前主题下的题目，根据当前状态归类到 part2 或 part3
            if current_topic is not None and current_state:
                if current_state == "part2":
                    current_topic["part2 questions"].append(line)
                elif current_state == "part3":
                    current_topic["part3 questions"].append(line)
            continue

    # 循环结束后保存最后一个主题
    if current_topic:
        if current_section == "part1":
            kb["part1"].append(current_topic)
        elif current_section == "part2&part3":
            kb["part2&part3"].append(current_topic)
    return kb


def generate_faiss_index(kb, dim=300):
    """
    将知识库中所有题目收集后生成向量（此处用随机向量模拟），构建 FAISS 索引。
    返回索引对象及所有题目列表。
    """
    all_questions = []
    for item in kb["part1"]:
        all_questions.extend(item.get("part1 questions", []))
    for item in kb["part2&part3"]:
        all_questions.extend(item.get("part2 questions", []))
        all_questions.extend(item.get("part3 questions", []))
    num_questions = len(all_questions)
    embeddings = np.random.random((num_questions, dim)).astype('float32')
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, all_questions


def main():
    txt_file = "ielts_speaking_questions1.txt"
    kb = parse_txt_to_kb(txt_file)
    with open("knowledge_base1.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=4)
    print("知识库已保存为 knowledge_base1.json")

    index, all_questions = generate_faiss_index(kb)
    faiss.write_index(index, "ielts_knowledge_base1.index")
    print("FAISS 索引已保存为 ielts_knowledge_base1.index")


if __name__ == "__main__":
    main()
