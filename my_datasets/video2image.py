import cv2
import os


# 打开视频文件
cap = cv2.VideoCapture('/home/nightbreeze/my_datasets/affordance_dataset/video/knife1.mp4')
folder_path = '/home/nightbreeze/my_datasets/affordance_dataset/video2image/knife1/'

# 如果文件夹不存在，则创建它
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")

# 定义帧计数器、计数器间隔和计数器
frame_count = 0
count_interval = 12
count = 0

# 读取视频帧
while True:
    ret, frame = cap.read()

    # 如果视频帧读取失败，则退出循环
    if not ret:
        break

    # 获取帧的宽度和高度，并计算长边和短边的比例
    height, width = frame.shape[:2]
    ratio = height / width if height > width else width / height

    # 如果长边像素超过1920，则限制长边为1920，短边按比例缩放
    long = 1440
    if height > long or width > long:
        if height > width:
            height = long
            width = int(height / ratio)
        else:
            width = long
            height = int(width / ratio)


    # 如果帧计数器是计数器间隔的倍数，则保存当前帧
    if frame_count % count_interval == 0:
        # 使用帧计数器作为图像文件名
        filename = 'knife1_' + str(count) + '.png'
        # 定义完整的文件路径
        file_path = folder_path + filename
        # 缩放图像
        frame = cv2.resize(frame, (width,height))
        # 保存当前帧
        cv2.imwrite(file_path, frame)
        # 增加计数器
        count += 1

    # 增加帧计数器
    frame_count += 1

# 释放视频文件和内存占用
cap.release()
cv2.destroyAllWindows()