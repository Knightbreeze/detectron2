import os
import random
import shutil

# 设置随机数种子，以保证每次运行程序得到的划分结果相同
random.seed()

# 定义数据集所在文件夹的路径和测试集、训练集文件夹的路径
dataset_dir = '/home/nightbreeze/deeplearning/detectron2/my_datasets/mydataset'
test_dir = '/home/nightbreeze/deeplearning/detectron2/my_datasets/test'
train_dir = '/home/nightbreeze/deeplearning/detectron2/my_datasets/train'

# 创建测试集、训练集文件夹
os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)

# 获取所有数据集文件名
file_names = [filename.split('.')[0] for filename in os.listdir(dataset_dir) if filename.endswith('.png')]

# 随机打乱文件名顺序
random.shuffle(file_names)

# 划分数据集
test_size = int(len(file_names) * 0.2)  # 测试集占总数据集的比例
test_file_names = file_names[:test_size]
train_file_names = file_names[test_size:]

# 将测试集数据复制到测试集文件夹中
for file_name in test_file_names:
    shutil.copy(os.path.join(dataset_dir, file_name + '.png'), os.path.join(test_dir, file_name + '.png'))
    shutil.copy(os.path.join(dataset_dir, file_name + '.json'), os.path.join(test_dir, file_name + '.json'))

# 将训练集数据复制到训练集文件夹中
for file_name in train_file_names:
    shutil.copy(os.path.join(dataset_dir, file_name + '.png'), os.path.join(train_dir, file_name + '.png'))
    shutil.copy(os.path.join(dataset_dir, file_name + '.json'), os.path.join(train_dir, file_name + '.json'))