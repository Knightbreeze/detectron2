from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import ColorMode

import math
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 用于从数据集中随机选择n个样本并将它们绘制出来，以便检查数据集的质量和样本的标注是否正确
def plot_samples(dataset_name, n, scale_size):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        im = cv2.imread(s["file_name"])
        v = Visualizer(im[:,:,::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        sample = v.get_image()[:, :, ::-1]
        width = int(sample.shape[1]*scale_size)
        height = int(sample.shape[0]*scale_size)
        dim = (width , height)
        sample = cv2.resize(sample, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Sample", sample)
        cv2.waitKey()
        cv2.destroyAllWindows()

# 设置训练参数
def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path)) # 模型初始化，指定使用那个预训练模型
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url) # 模型初始化，指定预训练模型的初始权重
    cfg.DATASETS.TRAIN = (train_dataset_name,) # 指定训练集的名称 
    cfg.DATASETS.TEST = (test_dataset_name,)   # 指定测试集的名称

    cfg.DATALOADER.NUM_WORKERS = 2    # 指定数据加载和读取的并发线程数（2条线运行） 
    cfg.SOLVER.IMS_PER_BATCH = 2      # 指定每个小批量训练的图像数量
    cfg.SOLVER.BASE_LR = 0.00025      # 指定学习率的初始值
    cfg.SOLVER.MAX_ITER = 2000        # 指定训练的最大迭代次数
    cfg.SOLVER.STEPS = []             # 指定学习率在训练过程中的变化步骤。这里将它设置为空列表，表示不使用学习率衰减策略。
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 模型中RoIHead的批量处理大小，即每张图片的RoIHead处理的样本数量,默认为512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # 指定模型需要检测的物体类别数量
    cfg.MODEL.DEVICE = device   # 指定模型运行的设备类型。这个参数的值可以是"cpu"或"cuda"
    cfg.OUTPUT_DIR = output_dir # 指定训练过程中的输出目录
    return cfg

# 定义绘制坐标系的函数
def draw_axes(image, center, length):
    x_axis = ((center[0],center[1]), (center[0]+length, center[1]))
    # print(image,'\n',x_axis)
    cv2.arrowedLine(image, x_axis[0], x_axis[1], (81,111,231), 3, tipLength=0.1)
    y_axis = ((center[0],center[1]), (center[0], center[1]-length))
    cv2.arrowedLine(image, y_axis[0], y_axis[1], (106,196,233), 3, tipLength=0.1)
    z_axis = ((center[0],center[1]), ((center[0]-int(length/math.sqrt(2))), (center[1]+int(length/math.sqrt(2)))))
    cv2.arrowedLine(image, z_axis[0], z_axis[1], (143,157,42), 3, tipLength=0.1)

    return image

# 显示预测的图片结果
def on_Image(image_path, predictor):
    # class_names  = ['contain'  , 'cut'        , 'grasp'  , 'grip'     , 'hammer'   , 'poke'   , 'wrap-grasp']
    # thing_colors = [(0,255,255), (255,255,255), (255,0,0), (255,0,255), (255,255,0), (0,0,255), (0,0,0)     ]
    class_names  = ['clamp', 'contain', 'cut', 'drill', 'grasp', 'poke', 'pound', 'wrap-grasp', 'hit', 'scoop']
    thing_colors = [(255,0,0), (255,255,0), (0,0,255)]
    # [(83,191,0), (249,199,79), (87,117,255), (14,171,124), (14,171,124), (249,65,68), (242,94,15), (0,221,255)]
    affordance_metadata = "affordance_metadata"
    MetadataCatalog.get(affordance_metadata).set(thing_classes = class_names,
                                          thing_colors = thing_colors,
                                          alpha = 0.5)
    metadata = MetadataCatalog.get("affordance_metadata")
    im = cv2.imread(image_path)
    # 改变image大小
    # scale_size = 5
    # width = int(im.shape[1]*scale_size)
    # height = int(im.shape[0]*scale_size)
    # dim = (width , height)
    # im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    
    outputs = predictor(im)

    # instance_mode:
    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """
   
    # 用于图像和标注可视化
    # im[:,:,::-1]：将图像数组中的BGR通道顺序转换为RGB顺序
    # metadata：指定数据集的元数据，其中'thing_classes'键包含类别名称列表
    # scale：指定可视化输出的缩放比例
    # instance_mode：指定可视化实例分割结果的颜色模式。ColorMode.IMAGE_BW，表示将实例分割结果绘制为黑白掩码。
    # print('--------------------','\n',outputs,'\n','--------------------')
    # v = Visualizer(im[:,:,::-1], metadata={'thing_classes':class_names}, scale=1, instance_mode = ColorMode.IMAGE_BW)
    v = Visualizer(im[:,:,::-1], metadata=metadata, scale=1, instance_mode=ColorMode.SEGMENTATION)
    # print('--------------------','\n',v,'\n','--------------------')
    # 将模型输出的Instances对象传递给Visualizer对象，并绘制实例分割结果。outputs["instances"]是模型输出的实例分割结果，to("cpu")将其转移到CPU上进行可视化。
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # print('--------------------','\n',out,'\n','--------------------')
    # 从Visualizer对象中获取图像结果，并将BGR通道顺序转换为RGB顺序
    result_onlymask = out.get_image()[:, :, ::-1]

    '''-------------------绘制中心点及坐标-------------------'''
    x_min = 100000000
    x_max = 0
    y_min = 100000000
    y_max = 0
    all_center = []
    class_name_list = []
    result = 0
    for i in range(len(outputs["instances"])):
        # 获取第i个实例的mask和bbox
        mask = outputs["instances"].pred_masks[i].cpu() 
        bbox = outputs["instances"].pred_boxes.tensor[i]

        # 计算mask的中心点坐标
        x1, y1, x2, y2 = bbox.int().tolist()
        # print(x1,'\n',y1,'\n',x2,'\n',y2)
        mask = mask[y1:y2, x1:x2]
        # print('--------------------','\n',mask,'\n','--------------------')
        ys, xs = np.where(mask)
        # print(xs,'----',ys,'\n')
        # print(xs.mean(),'----',ys.mean())
        center = np.array([x1+xs.mean(), y1+ys.mean()]).astype(int)
    
        # 输出中心点坐标
        class_idx = outputs["instances"].pred_classes[i].cpu().numpy()
        class_name = class_names[class_idx]
        all_center.append(center)
        class_name_list.append(class_name)

        # 输出中心点坐标和分类信息
        # print(f"Instance {i}:  center(x, y): ({center[0]}, {center[1]})  class: {class_name}")
        
        result = cv2.UMat(result_onlymask)
        result = result.get()

        

        # 绘制坐标系
        # length = 50
        # result = draw_axes(result, center, length)

        # # 在可视化结果上绘制中心点
        # radius = 5
        # color = (0, 0, 255)
        # thickness = -1
        # cv2.circle(result, (center[0], center[1]), radius, color, thickness)

        # 计算bbox
        if x_min>x1:
            x_min = x1
        if x_max<x2:
            x_max = x2
        if y_min>y1:
            y_min = y1
        if y_max<y2:
            y_max = y2
    allow = 75
    bbox = [x_min-allow, y_min-allow, x_max+allow, y_max+allow]    
    # print(f'min_point:({x_min},{y_min})','\n',f'max_point:({x_max},{y_max})')


    '''-------------------裁切图像-------------------'''
    # # 定义裁剪框大小
    # w, h = x_max - x_min + 20, y_max - y_min + 20

    # # 裁剪图像
    # crop_img = result_onlymask[y_min-10:y_max+10, x_min-10:x_max+10]

    # # 缩放图像
    # scale_factor = 512.0 / max(w, h)
    # resized_img = cv2.resize(crop_img, (int(w * scale_factor), int(h * scale_factor)))

    # # 对称式填补黑色像素
    # border_w = (512 - resized_img.shape[1]) // 2
    # border_h = (512 - resized_img.shape[0]) // 2
    # result_crop = cv2.copyMakeBorder(resized_img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    result_crop = result

    return all_center, bbox, result_onlymask, result, result_crop

# 调用摄像头进行实时检测
def on_Video(videoPath, predictor):
    class_names = ['clamp', 'contain', 'cut', 'drill', 'grasp', 'poke', 'pound', 'wrap-grasp', 'hit', 'scoop']
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened() == False):
        print("Error opening file...")
        return

    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:,:,::-1], metadata={'thing_classes':class_names}, scale=0.5 ,instance_mode = ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        # cv2.imread("Reuslt", output.get_image()[:,:,::-1])
        # cv2.namedWindow("result", 0)
        # cv2.resizeWindow("result", 1200, 600)

        #调用电脑摄像头进行检测
        cv2.namedWindow("result", cv2.WINDOW_FREERATIO) # 设置输出框的大小，参数WINDOW_FREERATIO表示自适应大小
        cv2.imshow("result" , output.get_image()[:,:,::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()

# # 调用Kinect Azure相机进行实时检测
# def on_Kinect_Azure_Video(predictor):
#     import pykinect_azure as pykinect

#     class_names  = ['contain'  , 'cut'        , 'grasp'  , 'grip'     , 'hammer'   , 'poke'   , 'wrap-grasp']
#     thing_colors = [(0,255,255), (255,255,255), (255,0,0), (255,0,255), (255,255,0), (0,0,255), (0,0,0)     ]
#     affordance_metadata = "affordance_metadata"
#     MetadataCatalog.get(affordance_metadata).set(thing_classes = class_names,
#                                           thing_colors = thing_colors,
#                                           alpha = 0.5)
#     metadata = MetadataCatalog.get("affordance_metadata")
#     # Initialize the library, if the library is not found, add the library path as argument
#     pykinect.initialize_libraries()

# 	# Modify camera configuration
#     device_config = pykinect.default_configuration
#     device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
# 	# print(device_config)

# 	# Start device
#     device = pykinect.start_device(config=device_config)

#     while True:

# 		# Get capture
#         capture = device.update()

# 		# Get the color image from the capture
#         ret, color_image = capture.get_color_image()

#         if not ret:
#             continue
			
# 		# Plot the image
#         predictions = predictor(color_image)
#         v = Visualizer(color_image[:,:,::-1], metadata, scale=0.5 ,instance_mode = ColorMode.SEGMENTATION)
#         output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

#         cv2.namedWindow("result", cv2.WINDOW_FREERATIO) # 设置输出框的大小，参数WINDOW_FREERATIO表示自适应大小
#         cv2.imshow("result" , output.get_image()[:,:,::-1])
# 		# Press q key to stop
#         if cv2.waitKey(1) == ord('q'): 
#             break
                
