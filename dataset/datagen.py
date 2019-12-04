#-*- coding:utf-8 -*-
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from config import config

class Tusimple(Dataset):
    def __init__(self, mode='train'): #mode:train/val/test
        self.data_path = config['Datapath']
        self.mode = mode
        self.image_sets, self.target_sets = self.loaddataset()

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, index):
        target = np.array(Image.open(self.target_sets[index]))
        img = np.array(Image.open(self.image_sets[index]))
        img = np.transpose(img, (2,0,1))
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        target = torch.from_numpy(target)
        return img, target

    def loaddataset(self):
        dataset_path = './namelists.txt'
        imagesets, targetsets = [], []
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split()
                imagesets.append(line[0])
                targetsets.append(line[1])
        return imagesets, targetsets

def get_data_loader(mode='train', num_workers=1, shuffle=True, batch_size=config['batch_size']):
    dataset = Tusimple(mode=mode)
    dataset_loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size,
                                num_workers=num_workers)
    return dataset_loader

#  def main():
    #  namefile = './out.txt'
    #  f = open(namefile,'a')
    #  labels_map = {0:0, 200:1, 204:1, 213:1, 209:1, 206:1, 207:1, 201:2, 203:2, 211:2, 208:2,
                    #  216:3, 217:3, 215:3, 218:4, 219:4, 210:5, 232:5, 214:6, 202:7, 220:7, 221:7,
                    #  222:7, 231:7, 224:7, 225:7, 226:7, 230:7, 228:7, 229:7, 233:7, 205:8, 212:8,
                    #  227:8, 223:8, 250:8, 249:0, 255:0}
    #  name = {0:'background', 1: 'dividing', 2:'guiding', 3:'stopping', 4:'chevron',
            #  5:'parking', 6:'zebra', 7:'thru/turn', 8:'reduction'}
    #  labels = [200, 204, 213, 209, 206, 207, 201, 203, 211, 208]
        #  # , 216, 217, 215, 218, 219, 210, 232, 214,
        #  #             202, 220, 221,222, 231, 224, 225, 226, 230, 228, 229, 233, 205, 212,227, 223, 250]
    #  test_dataloader = get_data_loader(mode='train', num_workers=1, shuffle=1, batch_size=config['batch_size'])
    #  for scans, targets in test_dataloader:
        #  img = targets[0]
        #  img_c = img.clone()
        #  im = Image.fromarray(np.uint8(img.numpy()))
        #  im.save('./t1.jpg')
        #  # im.show()
        #  one = torch.ones(img.shape)
        #  zero = torch.zeros(img.shape)
        #  zhi = torch.full(img.shape, 255)
        #  for i in labels:
            #  img_c = torch.where(img==i, one, img_c)
        #  img_c = torch.where(img_c==1, zhi, zero)
        #  img_c = Image.fromarray(np.uint8(img_c))
        #  # img.show()
        #  img_c.save('./t2.jpg')

        #  # img = targets.numpy()
        #  # img = img[0]
        #  # for i in range(img.shape[0]):
        #  #     for j in range(img.shape[1]):
        #  #        f.write(str(img[i][j])+' ')
        #  #     f.write('\n')
        #  # f.close()
        #  break

#  if __name__ == '__main__':
    #  main()
