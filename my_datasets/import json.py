import json

# 读取JSON文件
with open('/home/nightbreeze/deeplearning/detectron2/my_datasets/all_datasets/coffee_mug_1_1_1_crop.json', 'r') as f:
    data = json.load(f)

# 删除指定内容
for item in data:
    if item.get('label') == 'pourable' and item.get('shape_type') == 'polygon':
        data.remove(item)

# 写入修改后的JSON文件
with open('file.json', 'w') as f:
    json.dump(data, f, indent=4)