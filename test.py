from detectron2.engine import DefaultPredictor
import os
import cv2
import pickle
import re
from utils import *

cfg_save_path = "OD_cfg.pickle"
# 载入存储的配置文件，从pickle文件中加载Detectron2配置，并将其存储在cfg变量中
with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)
    # print(cfg)

# 将模型权重路径设置为模型输出目录中的model_final.pth文件
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 #设置模型预测时使用的置信度阈值
predictor = DefaultPredictor(cfg)


# '''测试文件夹中所有图片的affordance'''
# # 指定源文件夹和目标文件夹的路径
# id = 0
# source_dir = '/home/nightbreeze/newdisk/oneposeplus/demo/chuizi2-2/chuizi2-2-test/oldcolor_full'
# affmask_image_dir = '/home/nightbreeze/affmask_image'
# aff_information_dir = '/home/nightbreeze/aff_information'
# image_dir = '/home/nightbreeze/color_full'

# # 创建目标文件夹，如果它不存在
# if not os.path.exists(affmask_image_dir):
#     os.makedirs(affmask_image_dir)
# if not os.path.exists(aff_information_dir):
#     os.makedirs(aff_information_dir)
# if not os.path.exists(image_dir):
#     os.makedirs(image_dir)

# # 获取文件名数字
# def get_file_number(filename):
#     return int(re.findall('\d+', filename)[0])

# # 按照文件名数字从小到大的顺序排序
# filenames = sorted(os.listdir(source_dir), key=get_file_number)
# # 遍历源文件夹中的所有图片
# for filename in filenames:
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         # 加载图片
#         image_path = os.path.join(source_dir, filename)
#         aff_no_image = cv2.imread(image_path)
#         all_center, affbbox, result_onlymask, result_affmask, result_crop, class_name_list= on_Image(image_path, predictor, 1)
#         if 'grasp' in class_name_list and 'hammer' in class_name_list and len(class_name_list) == 2:
#             # print(class_name_list,type(class_name_list))
#             # 构造目标文件路径
#             affmask_path = os.path.join(affmask_image_dir, f"{id}.png")
#             aff_no_image_path = os.path.join(image_dir, f"{id}.png")
#             aff_information_path = os.path.join(aff_information_dir, f"{id}.txt")
#             # 保存affmask图片
#             cv2.imwrite(affmask_path, result_affmask)
#             cv2.imwrite(aff_no_image_path, aff_no_image)
#             # 保存affbbox和center的内容
#             save_str = ''
#             list_all_center = [coord for point in all_center for coord in point]
#             save_str = f"{' '.join(str(x) for x in affbbox)} {' '.join(str(x) for x in list_all_center)}"
#             save_file = open(aff_information_path, 'w')
#             save_file.write(save_str)
#             save_file.close()
#             print(id, all_center, affbbox)
#             # 更新已存在文件的序号
#             id += 1
# print('-----------------over!----------------')


# '''evaluation'''
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# from detectron2.data.datasets import register_coco_instances
# class_names = ['grasp', 'hammer', 'poke'] 
# test_dataset_name = "test_datasets"
# test_images_path = "my_datasets/test"
# test_json_annot_path = "my_datasets/test.json"
# register_coco_instances("test_datasets", {}, test_json_annot_path, test_images_path)
# MetadataCatalog.get("test_datasets").set(thing_classes = class_names,
#                                     evaluator_type = 'coco',
#                                     json_file=test_json_annot_path,
#                                     image_root=test_images_path)
# evaluator = COCOEvaluator("test_datasets", output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "test_datasets")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))
# # another equivalent way to evaluate the model is to use `trainer.test`

'''测试单张图片的affordance'''
image_path = "my_datasets/test/screwdriver2_45.png"
center, bbox, result_onlymask, result, result_crop = on_Image(image_path, predictor)
print('(x,y):',center,'\n','bbox(x1,y1,x2,y2):',bbox)
cv2.imshow("image", result)
cv2.waitKey()
cv2.destroyAllWindows()

'''连接Kinect Azure相机实时检测画面的affordance'''
# on_Kinect_Azure_Video(predictor)

'''连接电脑摄像头实时检测画面的affordance'''
# video_path = ""
# on_Video(0, predictor)   #0 表示调用电脑的摄像头来实时预测result = cv2.UMat(result)