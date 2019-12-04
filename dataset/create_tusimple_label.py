#-*- coding:utf-8 -*-
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from config import config
# import torch

def main():
    traindatapath = os.path.join(config['Datapath'], 'train_set')
    # train_path = os.path.join(traindatapath, 'clips')
    train_label = os.path.join(traindatapath, 'labels')

    count = 0
    for l in os.listdir(train_label):
        l_path = os.path.join(train_label, l)
        with open(l_path) as json_file:
            for line_json in json_file:
                print(count)
                line_json = line_json.strip()
                j_content = json.loads(line_json) #json.loads将字符串转换为字典，但是要求key必须使用双引号，不能用单引号
                # j_content = eval(line_json) #可以使用eval代替json.loads

                data_name = j_content['raw_file']
                data_name = data_name.split('/')
                data_name = '\\'.join(data_name)

                widths = j_content['lanes']
                height = j_content['h_samples']
                label_dot = []
                for i in range(len(widths)):
                    line = []
                    for j in range(len(height)):
                        if widths[i][j] != -2:
                            line.append([height[j],widths[i][j]])
                    if line:
                        label_dot.append(line)

                # data_path = os.path.join(traindatapath, data_name)
                # pic = Image.open(data_path)
                # datas_out_path = os.path.join(traindatapath, 'datas', str(count)+'.jpg')
                # pic.save(datas_out_path)

                img = np.zeros((720, 1280, 3))
                for i in range(len(label_dot)):
                    for j in label_dot[i]:
                        img[j[0]][j[1]]= np.array([255,255,255])

                img = Image.fromarray(np.uint8(img))
                # img.show()
                targets_out_path = os.path.join(traindatapath, 'targets', str(count)+'.jpg')
                img.save(targets_out_path)

                count += 1
                # for lane in label_dot:
                #     lane = list(zip(*lane))
                #     plt.scatter(lane[1], lane[0])

                # plt.imshow(pic)
                # plt.show()

def create_datapath_file():
    traindatapath = os.path.join(config['Datapath'], 'train_set')
    train_datas_path = os.path.join(traindatapath, 'datas')
    train_targets_path = os.path.join(traindatapath, 'targets')

    name_file = os.path.join(traindatapath, 'namelists.txt')
    filename = open(name_file, 'a')
    for f in os.listdir(train_datas_path):
        data_path = os.path.join(train_datas_path, f)
        target_path = os.path.join(train_targets_path, f)
        filename.write(data_path+' '+target_path+'\r')
    filename.close()

if __name__ == '__main__':
    main()
    # create_datapath_file()
