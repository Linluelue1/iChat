import os
import re
import shutil

source_dir = "output"
base_target_dir = "../ListeningChat/dataSet"  # 基础目录改为dataset

for filename in os.listdir(source_dir):
    if filename.endswith(".txt"):
        # 匹配格式：剑桥雅思编号-Test编号-Task编号答案解析.txt
        match = re.match(r"(\d+)-(\d+)-(\d+)听力原文\.txt", filename)
        if match:
            yl_num = match.group(1)  # 剑桥雅思编号
            test_num = match.group(2)  # Test编号
            task_num = match.group(3)  # Task编号

            # 构建目标路径：dataset/剑桥雅思{编号}/Test{编号}/Task{编号}
            target_dir = os.path.join(
                base_target_dir,
                f"剑桥雅思{yl_num}",  # 动态添加编号
                f"Test{test_num}",
                f"Task{task_num}"
            )

            os.makedirs(target_dir, exist_ok=True)
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(target_dir, filename)
            shutil.move(src_path, dest_path)
            print(f"Moved: {filename} -> {dest_path}")
        else:
            print(f"Skipped: {filename} (文件名格式不匹配)")