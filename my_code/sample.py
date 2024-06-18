from detectron2.data.datasets import register_coco_instances
from utils import *

# num_classes = 10                 # 需要检测的物体类别数量
class_names = ['clamp', 'contain', 'cut', 'drill', 'grasp', 'poke', 'pound', 'wrap-grasp', 'hit', 'scoop']    # 类别名（根据生成的train.json文件中的顺序填写）
train_dataset_name = "train_datasets"
train_images_path = "my_datasets/train"
train_json_annot_path = "my_datasets/train.json"

test_dataset_name = "test_datasets"
test_images_path = "my_datasets/test"
test_json_annot_path = "my_datasets/test.json"

# 注册训练集
register_coco_instances("train_datasets", {},train_json_annot_path, train_images_path)
MetadataCatalog.get("train_datasets").set(thing_classes = class_names,
                                    evaluator_type = 'coco',
                                    json_file=train_json_annot_path,
                                    image_root=train_images_path)

# 注册测试集
register_coco_instances("test_datasets", {}, test_json_annot_path, test_images_path)
MetadataCatalog.get("test_datasets").set(thing_classes = class_names,
                                    evaluator_type = 'coco',
                                    json_file=test_json_annot_path,
                                    image_root=test_images_path)

scale_size = 1         # 图片放大的倍数
sample_number = 5       # 随机选取样本的数量
plot_samples('train_datasets', sample_number, scale_size)
