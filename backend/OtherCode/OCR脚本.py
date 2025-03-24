import pdfplumber
import pytesseract
from PIL import Image
import re
import json
import os

# 确保安装了Tesseract-OCR，并将其路径添加到系统环境变量中
# 如果没有安装，可以从以下链接下载：https://github.com/tesseract-ocr/tesseract
# Windows用户可以使用预编译的安装包：https://github.com/UB-Mannheim/tesseract/wiki
# 安装完成后，确保将Tesseract的安装路径添加到系统的PATH环境变量中
# 显式指定 Tesseract-OCR 的路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def extract_ielts_data(pdf_path):
    data = {
        "article_structure": [],
        "key_vocabulary": {},
        "sentence_analysis": [],
        "question_analysis": [],
        "reference_translation": []
    }

    current_section = None
    current_paragraph = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 将PDF页面转换为图片
            page_image = page.to_image()
            image_path = f"temp_page_{page.page_number}.png"
            page_image.save(image_path, format="PNG")

            # 使用OCR提取文字
            text = pytesseract.image_to_string(Image.open(image_path), lang='chi_sim+eng')
            os.remove(image_path)  # 删除临时图片文件

            if not text:
                print(f"Warning: No text extracted from page {page.page_number}")
                continue

            # 过滤页码和页眉
            text = re.sub(r'www\.TopSage\.com\n\d+\n', '', text)
            text = re.sub(r'===== Page \d+ =====\n', '', text)

            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # 识别章节标题
                if re.match(r'篇章结构\s*', line):
                    current_section = "article_structure"
                    print(f"Detected section: {current_section}")
                    continue
                elif re.match(r'必背词汇\s*', line):
                    current_section = "key_vocabulary"
                    current_paragraph = 1
                    data["key_vocabulary"][current_paragraph] = []
                    print(f"Detected section: {current_section}")
                    continue
                elif re.match(r'难句解析\s*', line):
                    current_section = "sentence_analysis"
                    print(f"Detected section: {current_section}")
                    continue
                elif re.match(r'试题分析\s*', line):
                    current_section = "question_analysis"
                    print(f"Detected section: {current_section}")
                    continue
                elif re.match(r'参考译文\s*', line):
                    current_section = "reference_translation"
                    print(f"Detected section: {current_section}")
                    continue

                # 处理不同章节内容
                if current_section == "article_structure":
                    if re.match(r'体裁', line):
                        data["article_structure"].append(line.strip())
                elif current_section == "key_vocabulary":
                    if re.match(r'第\d+段\s*', line):
                        current_paragraph = int(re.search(r'\d+', line).group())
                        data["key_vocabulary"][current_paragraph] = []
                    else:
                        # 提取词汇（中英混合）
                        items = re.findall(r'([a-zA-Z\s]+?)\s*([\u4e00-\u9fff].*?)(?=\n|$)', line)
                        for en, zh in items:
                            data["key_vocabulary"][current_paragraph].append({
                                "term": en.strip(),
                                "explanation": zh.strip()
                            })
                # 其他章节的处理逻辑...

    return data

# 使用示例
pdf_path = "../dataSet/T4S1Reading.pdf"
result = extract_ielts_data(pdf_path)

# 保存为JSON
with open("ielts_knowledge_base.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("数据提取完成并已保存为ielts_knowledge_base.json")