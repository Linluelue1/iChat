import os
import re
import shutil


def organize_files(source_dir="dataSet", action='move'):
    """
    将文件按剑桥雅思版本分级存储
    :param source_dir: 源目录路径
    :param action: 文件操作方式 'move' 或 'copy'
    """
    # 验证操作类型
    if action not in ('move', 'copy'):
        raise ValueError("操作类型必须为 'move' 或 'copy'")

    # 遍历源目录
    for filename in os.listdir(source_dir):
        # 仅处理txt文件
        if not filename.endswith(".txt"):
            continue

        # 匹配文件名格式（例如：4-1-1.txt）
        match = re.match(r"^(\d+)-(\d+)-(\d+)\.txt$", filename)
        if not match:
            print(f"跳过无效文件名：{filename}")
            continue

        # 解析文件信息
        ver, test_num, task_num = match.groups()

        # 构建目标路径（例如：剑桥雅思4/Test1/Task1）
        target_dir = os.path.join(
            source_dir,
            f"剑桥雅思{ver}",
            f"Test{test_num}",
            f"Task{task_num}"
        )

        # 创建目录（exist_ok=True表示目录存在时不报错）
        os.makedirs(target_dir, exist_ok=True)

        # 构建完整文件路径
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(target_dir, filename)

        try:
            # 执行文件操作
            if action == 'move':
                shutil.move(src_path, dest_path)
                print(f"移动文件：{filename} → {dest_path}")
            else:
                shutil.copy(src_path, dest_path)
                print(f"复制文件：{filename} → {dest_path}")
        except Exception as e:
            print(f"操作失败：{filename} ({str(e)})")


if __name__ == "__main__":
    # 使用示例（默认移动文件）
    organize_files(action='move')  # 改为'copy'则保留原文件

    """最终目录结构：
    dataSet/
    ├── 剑桥雅思4/
    │   ├── Test1/
    │   │   └── Task1/
    │   │       └── 4-1-1.txt
    │   └── Test2/
    │       └── Task2/
    │           └── 4-2-2.txt
    └── 剑桥雅思5/
        └── Test1/
            └── Task1/
                └── 5-1-1.txt
    """