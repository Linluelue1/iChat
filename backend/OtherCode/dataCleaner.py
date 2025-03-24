def clean_file_content(filename, remove_str):
    """清理文本文件中包含特定字符串的行"""
    try:
        # 读取原始内容
        with open(filename, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        # 过滤不需要的行
        cleaned_lines = [line for line in original_lines if remove_str not in line]

        # 写入清理后的内容
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

        print(f"成功清理文件：{filename}")
        print(f"原始行数：{len(original_lines)} → 清理后行数：{len(cleaned_lines)}")
        print(f"删除包含字符串 [{remove_str}] 的行：{len(original_lines) - len(cleaned_lines)}")

    except Exception as e:
        print(f"处理文件时出错：{str(e)}")


# 使用示例（直接修改原文件）
clean_file_content(
    filename='../ListeningChat/听力答案合集.txt',
    remove_str='听力答案解析'
)