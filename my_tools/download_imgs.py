import os 
import urllib.request as urlrequest
import multiprocessing as mp 
import json
import sys

def download(image_path):
    save_root = '/opt/tiger/minist/datasets/groot_voc/JPEGImages'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    try_times = 0
    while try_times < 5:
        try:
        # if image_path.startswith('http://') or image_path.startswith('https://'):
            resp = urlrequest.urlopen(image_path, timeout=10)
            data = resp.read()
            
            save_path = image_path.split('/')[-1]
            # print(save_path)
            if not save_path.endswith('.jpg') :
                save_path = save_path + '.jpg'
            fp = open(os.path.join(save_root, save_path),'wb')
            fp.write(data)
            fp.close()
            break
        except Exception:
            try_times += 1


if __name__ == "__main__":
    count = 0
    url_lst = []
    file = '/opt/tiger/minist/data_process/res_data/new_img_in_val.txt'
    with open(file) as f:
        for line in f:
            json_line = json.loads(line)
            img_url = json_line['image_url']
            # print(img_url)
            # count = count + 1
            # if count == 10:
            #     break
            # url, label = line 
            url_lst.append(img_url)

    print('images nums : {}'.format(len(url_lst)))
    # print(url_lst[:3])
    # sys.exit()
    num_workers = 30
    p = mp.Pool(num_workers)
    p.map(download, url_lst)

