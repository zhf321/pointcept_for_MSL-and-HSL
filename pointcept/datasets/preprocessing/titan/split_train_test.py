import os
import shutil
from tqdm import tqdm

# 原始文件夹和目标文件夹
source_folder = '/home/zhaohaifeng/data/titan_pth/titan_512'  # 替换为实际的文件夹路径
output_folder = source_folder  # 替换为实际的文件夹路径

# 创建训练集(train)和验证集(val)文件夹
train_folder = os.path.join(output_folder, 'train')
val_folder = os.path.join(output_folder, 'val')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# 获取源文件夹中所有文件
# all_files = sorted(os.listdir(source_folder))

# 获取源文件夹中所有以.pth结尾的文件
all_files = [file_name for file_name in os.listdir(source_folder) if file_name.endswith('.pth')]
all_files = sorted(all_files)

# 按照一定规则将文件分配到不同的文件夹
for i, file_name in tqdm(enumerate(all_files), total=len(all_files), desc="Copying files"):
    source_path = os.path.join(source_folder, file_name)
    if i % 4 == 0:  # 每隔4个文件放到val文件夹
        destination_path = os.path.join(val_folder, file_name)
    else:
        destination_path = os.path.join(train_folder, file_name)

    # 复制文件
    shutil.move(source_path, destination_path)
    
    # print(f"{file_name}已移动！")

print("文件分配完成。")
