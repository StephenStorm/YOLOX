import xml.etree.ElementTree as ET
import cv2
import json
import os
from lxml.etree import Element, SubElement, tostring
# from xml.dom.minidom import parseString
import urllib.request as urlrequest
import numpy as np
import multiprocessing as mp 



class_map = {0:'beasts', 1:'bird', 2:'fish', 3:'insect', 4:'plant', 5:'person'}

def get_anno(line):
    json_line = json.loads(line)
    img_url = json_line['image_url']
    bbox = json_line['bboxes']
    labels = json_line['labels']
    save_xml(img_url, bbox, labels)


def save_xml(image_path, bbox, label, annotion_dir='/opt/tiger/minist/datasets/groot/Annotations', img_dir = '/opt/tiger/minist/datasets/groot/JPEGImages'):
    '''
    将CSV中的一行
    000000001.jpg [[1,2,3,4],...]
    转化成
    000000001.xml

    :param image_name:图片名
    :param bbox:对应的bbox
    :param save_dir:
    :param width:这个是图片的宽度，博主使用的数据集是固定的大小的，所以设置默认
    :param height:这个是图片的高度，博主使用的数据集是固定的大小的，所以设置默认
    :param channel:这个是图片的通道，博主使用的数据集是固定的大小的，所以设置默认
    :return:
    '''
    
    resp = urlrequest.urlopen(image_path, timeout=10)
    data = resp.read()
    
    # get img shape
    bina = np.frombuffer(data, dtype='uint8')
    image = cv2.imdecode(bina, cv2.IMREAD_COLOR)
    height, width, channel = image.shape
    # print(type(data))

    image_name = image_path.split('/')[-1]
    if not image_name.endswith('.jpg') :
            image_name = image_name + '.jpg'

    fp = open(os.path.join(img_dir, image_name),'wb')
    fp.write(data)
    fp.close()

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    # node_folder.text = 'JPEGImages'
    node_folder.text = 'GROOT'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    for i in range(len(bbox)):
        left, top, right, bottom = bbox[i]
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = class_map[label[i]]
        node_pose = SubElement(node_object, "pose")
        node_pose.text = str(0)
        node_pose = SubElement(node_object, "truncated")
        node_pose.text = str(0)
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom

    xml = tostring(node_root, pretty_print=True)  
    # dom = parseString(xml)

    save_xml = os.path.join(annotion_dir, image_name[:-4] + '.xml')
    with open(save_xml, 'wb') as f:
        f.write(xml)

    return


if __name__ == "__main__":
    # line_list = []
    json_file = '/opt/tiger/minist/datasets/groot/wy_val_plant-retag-nopeople.json'
    with open(json_file, 'r') as jf:
        line_list = jf.readlines()
    
    print('img num: {}'.format(len(line_list)))
    num_workers = 40
    p = mp.Pool(num_workers)
    p.map(get_anno, line_list)