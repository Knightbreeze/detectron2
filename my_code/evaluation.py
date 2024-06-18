import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

from detectron2.data import MetadataCatalog, DatasetCatalog

# evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from utils import get_train_cfg
from detectron2.engine import DefaultTrainer
import os
import pickle

config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"  # 指定使用那个预训练模型
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"   # 指定使用那个预训练模型初始权重

output_dir = "./output/number_detection" # 训练过程的输出目录

num_classes = 10                 # 需要检测的物体类别数量
class_names = ['clamp', 'contain', 'cut', 'drill', 'grasp', 'poke', 'pound', 'wrap-grasp', 'hit', 'scoop']   # 类别名（根据生成的train.json文件中的顺序填写）

device = "cuda"                 # 使用GPU进行训练

train_dataset_name = "train_datasets"
train_images_path = "my_datasets/train"
train_json_annot_path = "my_datasets/train.json"

test_dataset_name = "test_datasets"
test_images_path = "my_datasets/test"
test_json_annot_path = "my_datasets/test.json"

cfg_save_path = "OD_cfg.pickle" # 指定保存Detectron2配置对象的文件路径和文件名

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

# MetadataCatalog.get("test")

cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

    
# 将Detectron2配置对象保存为pickle格式的文件
with open(cfg_save_path, 'wb') as f:
    pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
    
 # 创建一个目录，用于保存Detectron2训练期间产生的日志、权重文件和其他中间文件
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 创建一个默认的Detectron2训练器，通过传入Detectron2配置对象cfg来配置训练器
trainer = DefaultTrainer(cfg)

cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TEST = ("test_datasets",)
cfg.MODEL.WEIGHTS = os.path.join(output_dir, "final_model.pth") 
print(output_dir)


evaluator = COCOEvaluator("test_datasets", ("bbox", "segm"), False, output_dir=output_dir)
val_loader = build_detection_test_loader(cfg, "test_datasets")
print(inference_on_dataset(trainer.model, val_loader, evaluator))