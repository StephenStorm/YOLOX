import os
root = '/opt/tiger/minist/datasets/groot_voc/Annotations/'
os.chdir(root)
list_file = '/opt/tiger/minist/tools/todo.txt'
with open(list_file, 'r') as f:
    for line in f:
        line = line.strip()
        new_name = line[:-4].replace("xml", "jpg") + line[-4:]
        os.rename(line, new_name)
        