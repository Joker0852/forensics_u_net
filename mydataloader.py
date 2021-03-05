import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchvision.transforms as transforms
import os

img_transform = transforms.Compose([transforms.ToTensor()])
label_transform = transforms.Compose([transforms.ToTensor()])

def load_image(path):
    img = cv2.imread(path)
    return img


def load_label(path):
    label = cv2.imread(path, 0)
    return label


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_datasetdir(dir):
    img_paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in os.walk(dir):
        fnames = sorted(fnames)
        for index in range(0, len(fnames)):
            fname = fnames[index]
            if is_image_file(fname):
                img_paths.append(fname)
    return img_paths


def onehot(data, n):
    b = data[:, :, np.newaxis].astype(np.float32) / 255
    buf = np.concatenate((b,1.0 - b), axis=2)
    return buf

def label_norm(data):
    buf = data / 255
    return buf

class mydataload(Dataset):
    def __init__(self, opt, img_transform=img_transform, label_transform=label_transform,
                 origin_transform=origin_transform,load_image=load_image, load_label=load_label):
        super(Mydataload, self).__init__()
        self.opt = opt
        fh = make_datasetdir(self.opt.img_path)
        imgs = []

        for line in fh:  # 迭代该列表#按行循环列表中内容
            line = line.strip('\n')
            image_fullpath = self.opt.img_path + line
            label_fullpath = self.opt.label_path + line
            image_path = line

            imgs.append((image_fullpath, label_fullpath,line))  # 组成tuple 加入list

        self.imgs = imgs
        self.img_transform = img_transform
        self.lab_transform = label_transform

        self.img_loader = load_image
        self.lab_loader = load_label

    # 使用__getitem__()对数据进行预处理并返回想要的信息
    def __getitem__(self, index):
        image_path, label_path, origin_path, path = self.imgs[index]
        img_file = self.img_loader(image_path)
        label_file = self.lab_loader(label_path)
        lab_file = label_norm(label_file)
        # label_file = onehot(label_file, 2).astype(np.float32)

        if self.img_transform is not None:
            # 数据标签转换为tensor
            img = self.img_transform(img_file)
        if self.lab_transform is not None:
            label = self.lab_transform(label_file)

        return {'img': img, 'lab': label, 'img_path': path}

    def __len__(self):
        return len(self.imgs)  # 总图片数/batch_size（向上取整）

