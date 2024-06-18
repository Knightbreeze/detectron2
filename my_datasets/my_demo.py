# import torch, detectron2
# # !nvcc --version
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import requests

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

response = requests.get('http://images.cocodataset.org/val2017/000000439715.jpg')
open("input.jpg", "wb").write(response.content)

im = cv2.imread("./input.jpg")
cv2.imshow('demo', im)
cv2.waitKey()
cv2.destroyAllWindows()

# 加载config配置
cfg = get_cfg()
# 导入主干网络为 mask rcnn resnet- 的配置文件
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
# 为模型增加阈值为0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# 如果使用CPU而不是GPU，则使用下一行
# cfg.MODEL.DEVICE = "cpu"

# 用这个cfg构建一个预测器，并在输入上运行它
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
# 查看输出
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
# 在Visualizer上绘制预测的image
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("image", out.get_image()[:, :, ::-1])
cv2.waitKey()
cv2.destroyAllWindows()