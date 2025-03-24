def remove_lines_above_separator(filename):
    """删除分割线上方四行的无效数据"""
    try:
        # 读取文件内容
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 查找所有分隔符行索引
        separator = "===== Page Separator ====="
        split_indices = [
            i for i, line in enumerate(lines)
            if line.strip() == separator
        ]

        # 收集需要删除的行号（使用集合避免重复）
        to_delete = set()
        for idx in split_indices:
            # 计算前四行的范围
            start = max(0, idx - 1)  # 处理文件开头的情况
            end = idx - 1
            if end >= start:  # 有效范围时才添加
                to_delete.update(range(start, end + 1))

        # 保留未标记删除的行
        cleaned_lines = [
            line for i, line in enumerate(lines)
            if i not in to_delete
        ]

        # 覆盖写入原文件（建议提前备份）
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

        print(f"处理完成！共删除 {len(lines)-len(cleaned_lines)} 行无效数据")
        print(f"原始行数：{len(lines)} → 当前行数：{len(cleaned_lines)}")

    except Exception as e:
        print(f"处理文件时出错：{str(e)}")

# 使用示例（直接修改原文件）
remove_lines_above_separator('../SpeakingChat/ielts_speaking_questions1.txt')