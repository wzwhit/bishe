#-*- coding:utf-8 -*-
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from config import config

class Baidu_Datasets(Dataset):
    def __init__(self): #mode:train/val/test
        self.image_sets, self.target_sets = self.loaddataset()
        self.nums = len(self.image_sets)
        self.labels = [200, 204, 213, 209, 206, 207, 201, 203, 211, 208]

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

    def create_labels(self):
        for index in range(self.nums):
            print(index)
            img = np.array(Image.open(self.image_sets[index]))

            target = np.array(Image.open(self.target_sets[index]))
            target = torch.from_numpy(target).float()
            target_c = target.clone()

            one = torch.ones(target.shape)
            zero = torch.zeros(target.shape)
            zhi = torch.full(target.shape, 255)
            for i in self.labels:
                target_c = torch.where(target == i, one, target_c)
            target_c = torch.where(target_c == 1, zhi, zero)

            img = Image.fromarray(np.uint8(img))
            target_c = Image.fromarray(np.uint8(target_c.numpy()))
            datapath = os.path.join(config['Datapath'],'datas',str(index)+'.jpg')
            labelpath = os.path.join(config['Datapath'], 'labels', str(index) + '.jpg')
            img.save(datapath)
            target_c.save(labelpath)

def main():
    dataset = Baidu_Datasets()
    dataset.create_labels()

if __name__ == '__main__':
    main()
