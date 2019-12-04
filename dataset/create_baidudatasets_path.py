#-*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image
from config import config

def main():
    train_datas_path = os.path.join(config['Datapath'], 'Image_Data')
    train_labels_path = os.path.join(config['Datapath'], 'Gray_Label')

    name_file = './namelists.txt'
    filename = open(name_file, 'a')
    for f in os.listdir(train_datas_path):
        f_data_path = os.path.join(train_datas_path, f, 'ColorImage')
        f_label_path = os.path.join(train_labels_path, 'Label_'+f, 'Label')
        for ff in os.listdir(f_data_path):
            ff_data_path = os.path.join(f_data_path, ff)
            ff_label_path = os.path.join(f_label_path, ff)
            for fff in os.listdir(ff_data_path):
                fff_data_path = os.path.join(ff_data_path, fff)
                fff_label_path = os.path.join(ff_label_path, fff)
                for ffff in os.listdir(fff_data_path):
                    image = os.path.join(fff_data_path, ffff)
                    label = os.path.join(fff_label_path, ffff[:-4]+'_bin.png')
                    #  print(image, '   ',label)
                    filename.write(image+' '+label+'\n')
    filename.close()


if __name__ == '__main__':
    main()
