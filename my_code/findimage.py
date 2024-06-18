import os
import shutil

def copy_images(source_folder, target_folder):
    counter = 0
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file == 'results.png':
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_folder, str(counter) + '.png')
                shutil.copyfile(source_path, target_path)
                counter += 1
    print('共复制了%d个文件' % counter)

if __name__ == '__main__':
    source_folder = '/home/nightbreeze/deeplearning/onepose/OnePose/runs/vis/GATsSPG/0758-shouzuan222_shouzuan222-2'
    target_folder = '/home/nightbreeze/onepose_image/0758'
    copy_images(source_folder, target_folder)