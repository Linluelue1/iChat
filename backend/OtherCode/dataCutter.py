import re
import os

CN_NUM = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
}


def mixed2num(s):
    """混合数字转换（兼容单个中文字符或阿拉伯数字）"""
    if s.isdigit():
        return int(s)
    return CN_NUM.get(s, 0)


def split_and_save(content, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # 分割区块
    sections = re.split(r"=+\n+", content)

    for section in sections:
        if not section.strip():
            continue

        # 模式1：匹配 TestXSectionY 格式（核心改进）
        match = re.search(
            r"内容:\n剑(\d+)[\s-]*"
            r"test [\s-]*([一二三四五六七八九十\d]+)[\s\S]*?"
            r"Section [\s-]*([一二三四五六七八九十\d]+)",
            section,
            flags=re.MULTILINE
        )
        if not match:
            # 模式2：匹配中文套数描述（旧格式）
            match = re.search(
                r"内容:\n剑桥雅思(\d+)听力第([一二三四五六七八九十\d]+)套题目第([一二三四五六七八九十\d]+)部分",
                section,
                flags=re.MULTILINE
            )

        if match:
            ver = match.group(1)
            test_num = mixed2num(match.group(2))
            section_num = mixed2num(match.group(3))

            filename = f"{ver}-{test_num}-{section_num}听力原文.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(section.strip())
            print(f"生成文件：{filepath}")
        else:
            print(f"忽略未匹配区块：{section[:50]}...")


# 测试用例验证
test_cases = [
    "内容：\n剑桥雅思4 Test 2听力Section 3答案解析 Details of assignment",  # → 4-2-3.txt
    "内容：\n剑桥雅思10Test3Section2听力答案解析 Dolphin Conservation Trust",  # → 10-3-2.txt
    "内容：\n剑桥雅思12Test6Section1听力答案解析 Events during Kenton Festival",  # → 12-6-1.txt
    "内容：\n剑桥雅思17听力第三套题目第二部分",  # → 17-3-2.txt
]

if __name__ == "__main__":
    with open("../ListeningChat/听力原文合集.txt", "r", encoding="utf-8") as f:
        content = f.read()
    split_and_save(content)