def remove_specific_string(filename, target_str):
    """删除文件中所有出现的指定字符串（保留该行其他内容）"""
    try:
        # 读取原始内容
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # 执行字符串替换
        modified_content = content.replace(target_str, '')

        # 计算删除次数
        original_count = content.count(target_str)
        new_count = modified_content.count(target_str)
        removed_count = original_count - new_count

        # 写入修改后的内容
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(modified_content)

        print(f"成功处理文件：{filename}")
        print(f"共删除 {removed_count} 处目标字符串：{target_str}")

    except Exception as e:
        print(f"处理文件时出错：{str(e)}")

# 使用示例（直接修改原文件）
remove_specific_string(
    filename='../SpeakingChat/ielts_speaking_questions1.txt',
    target_str='this is from Laokaoya website. '
)