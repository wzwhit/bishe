import torch
import argparse
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random
from torch.utils.data import DataLoader

def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass


def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass

class Cityscapes(torch.utils.data.Dataset):
    def __init__(self, image_path, csv_path, scale, loss='dice', mode='train'):
        super().__init__()
        self.mode = mode
        self.image_list, self.label_list = self._get_city_pairs(image_path, self.mode)

        self.fliplr = iaa.Fliplr(0.5)
        self.label_info = get_label_info(csv_path)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        # self.crop = transforms.RandomCrop(scale, pad_if_needed=True)
        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])
        img = transforms.Resize((self.image_size[0],self.image_size[1]), Image.BILINEAR)(img)
        # random crop image
        # =====================================
        # w,h = img.size
        # th, tw = self.scale
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)
        # img = F.crop(img, i, j, th, tw)
        # =====================================

        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # randomly resize image and random crop
        # =====================================
        if self.mode == 'train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        # =====================================

        img = np.array(img)

        # load label
        label = Image.open(self.label_list[index])
        label = transforms.Resize((self.image_size[0],self.image_size[1]), Image.BILINEAR)(label)

        # crop the corresponding label
        # =====================================
        # label = F.crop(label, i, j, th, tw)
        # =====================================

        # randomly resize label and random crop
        # =====================================
        if self.mode == 'train':
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        label = np.array(label)[:,:,:3]


        # augment image and label
        if self.mode == 'train':
            seq_det = self.fliplr.to_deterministic()
            img = seq_det.augment_image(img)
            label = seq_det.augment_image(label)


        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        #  if self.loss == 'dice':
            #  # label -> [num_classes, H, W]
            #  label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)

            #  label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            #  # label = label.astype(np.float32)
            #  label = torch.from_numpy(label)

            #  return img, label

        #  elif self.loss == 'crossentropy':
            #  label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            #  # label = label.astype(np.float32)
            #  label = torch.from_numpy(label).long()
        label_dice = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)
        label_dice = np.transpose(label_dice, [2, 0, 1]).astype(np.float32)
        label_dice = torch.from_numpy(label_dice)

        label_ce = one_hot_it_v11(label, self.label_info).astype(np.uint8)
        label_ce = torch.from_numpy(label_ce).long()

        return img, label_dice, label_ce

    def __len__(self):
        return len(self.image_list)

    def _get_city_pairs(self, folder, split='train'):
        def get_path_pairs(img_folder, mask_folder):
            img_paths = []
            mask_paths = []
            for root, _, files in os.walk(img_folder):
                for filename in files:
                    if filename.endswith(".png"):
                        imgpath = os.path.join(root, filename)
                        foldername = os.path.basename(os.path.dirname(imgpath))
                        maskname = filename.replace('leftImg8bit', 'gtFine_color')
                        maskpath = os.path.join(mask_folder, foldername, maskname)
                        if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                            img_paths.append(imgpath)
                            mask_paths.append(maskpath)
                        else:
                            print('cannot find the mask or image:', imgpath, maskpath)
            print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
            return img_paths, mask_paths

        if split in ('train', 'val'):
            img_folder = os.path.join(folder, 'leftImg8bit/' + split)
            mask_folder = os.path.join(folder, 'gtFine/' + split)
            img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
            return img_paths, mask_paths
        else:
            assert split == 'trainval'
            print('trainval set')
            train_img_folder = os.path.join(folder, 'leftImg8bit/train')
            train_mask_folder = os.path.join(folder, 'gtFine/train')
            val_img_folder = os.path.join(folder, 'leftImg8bit/val')
            val_mask_folder = os.path.join(folder, 'gtFine/val')
            train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
            val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
            img_paths = train_img_paths + val_img_paths
            mask_paths = train_mask_paths + val_mask_paths
        return img_paths, mask_paths


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')

    args = parser.parse_args(params)

    # create dataset and dataloader
    csv_path = os.path.join(args.data, 'class_dict.csv')
    dataset_train = Cityscapes(args.data, csv_path, scale=(args.crop_height, args.crop_width),
                           loss=args.loss, mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    for i, (data, label) in enumerate(dataloader_train):
        print(i, data.shape, label.shape)


if __name__ == '__main__':
    params = [
        '--data', 'G:\\DATASETS\\Cityscapes',
        '--num_workers', '8',
        '--batch_size', '2',  # 6 for resnet101, 12 for resnet18
    ]
    main(params)

