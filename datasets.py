import random
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from scipy.io import loadmat
from torchvision import transforms

class DHFMDataset(Dataset):
    def __init__(self, args, transform=None, train=True):
        super().__init__()

        # filepath = './MNIST_MAT/MNIST_DHFM_Feature.mat
        filepath = './MNIST_MAT/MNIST_DHFM_Fea_S07.mat'
        data = loadmat(filepath)

        # train_x, val_x = data['Fea_Train_Arr'], data['Fea_Test_Arr']  # (60000, 106)  (10000, 106)
        train_x, val_x = data['Fea_Train_Arr'], data['Fea_Test_Rot_Arr']  # (60000, 106)  (10000, 106)

        label_train = './MNIST_MAT/Mnist_Train_Label.mat'
        label_val = './MNIST_MAT/Mnist_Test_Label.mat'
        train_y = loadmat(label_train)
        val_y = loadmat(label_val)
        train_y, val_y = train_y['Train_Label'][:, 0], val_y['Test_Label'][:, 0]  # (60000, )  (10000, )

        train_x = torch.from_numpy(train_x).type(torch.float32)
        val_x = torch.from_numpy(val_x).type(torch.float32)
        train_y = torch.from_numpy(train_y).type(torch.long)
        val_y = torch.from_numpy(val_y).type(torch.long)

        self.train_x, self.val_x = train_x, val_x
        self.train_y, self.val_y = train_y, val_y

        self.train = train
        self.transform = transform

    def __len__(self):
        if self.train:
            return len(self.train_x)
        return len(self.val_x)

    def load_feature(self, index):
        if self.train:
            fea = self.train_x[index]
            target = self.train_y[index]
        else:
            fea = self.val_x[index]
            target = self.val_y[index]
        return fea, target


    def __getitem__(self, index):
        fea, target = self.load_feature(index)
        # fea = transforms.ToTensor()(fea)
        return fea, target


class Mnist1dDataset(Dataset):
    def __init__(self, args, transform=None, train=True):
        super().__init__()
        mnist_img = './MNIST_MAT/Mnist_Train_Imgs.mat'
        mnist_label = './MNIST_MAT/Mnist_Train_Label.mat'
        val_img = './MNIST_MAT/Mnist_Test_Imgs.mat'
        val_label = './MNIST_MAT/Mnist_Test_Label.mat'
        train_x = loadmat(mnist_img)['Train_Img_Arr']  # (28, 28, 60000)
        train_y = loadmat(mnist_label)['Train_Label'][:, 0]  # (60000,)

        # mnist_rot = './MNIST_MAT/Mnist_Test_Rot_Imgs.mat'
        # val_x = loadmat(mnist_rot)['Test_Rot_Img_Arr']

        val_x = loadmat(val_img)['Test_Img_Arr']
        val_y = loadmat(val_label)['Test_Label'][:, 0]

        train_x = torch.from_numpy(train_x).type(torch.float32).permute(2, 0, 1).reshape(60000, 784)
        val_x = torch.from_numpy(val_x).type(torch.float32).permute(2, 0, 1).reshape(10000, 784)
        train_y = torch.from_numpy(train_y).type(torch.long)
        val_y = torch.from_numpy(val_y).type(torch.long)

        self.train_x, self.val_x = train_x, val_x
        self.train_y, self.val_y = train_y, val_y

        self.train = train
        self.transform = transform

    def __len__(self):
        if self.train:
            return len(self.train_x)
        return len(self.val_x)

    def load_feature(self, index):
        if self.train:
            fea = self.train_x[index]
            target = self.train_y[index]
        else:
            fea = self.val_x[index]
            target = self.val_y[index]
        return fea, target


    def __getitem__(self, index):
        fea, target = self.load_feature(index)
        # fea = transforms.ToTensor()(fea)
        return fea, target


default_transform = transforms.Compose([
    transforms.ToTensor(),  # # to tensor, to float(0, 1)
])


