import os


save_root = '/opt/tiger/minist/datasets/groot/JPEGImages/'
val_file = '/opt/tiger/minist/datasets/groot/ImageSets/Main/val.txt'
img_name = os.listdir(save_root)
img_name = [img[:-4]+'\n' for img in img_name]
val = open(val_file, 'w')
val.writelines(img_name)
val.close()