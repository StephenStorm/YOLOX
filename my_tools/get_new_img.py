import os 
import json
import sys
from tqdm import tqdm
img_folder = '/opt/tiger/minist/datasets/groot_voc/JPEGImages'

exist_img = os.listdir(img_folder)
print('exist_img nums {}'.format(len(exist_img)))
exist_img = [img[:-4] for img in exist_img]
# print(exist_img[:5])

new_img_path = '/opt/tiger/minist/data_process/res_data/groot_det_train_zhanglb_20210816after_quality_filter.json'
img_path = '/opt/tiger/minist/data_process/res_data/train_new.txt'

count = 0
count2 = 0
with open(new_img_path, 'r') as f, open(img_path, 'w') as w:
    for line in tqdm(f):
        json_line = json.loads(line.strip())
        img_url = json_line['image_url'].split('.jpg')[0]
        # print(img_url)
        img_name = img_url.split('/')[-1]
        # print(img_name)
        # sys.exit()
        if img_name in exist_img:
            count = count + 1
        else :
            count2 = count2 + 1
            w.write(line)

print(count, )