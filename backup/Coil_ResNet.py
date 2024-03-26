import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as Data

import torchvision
import torchvision.transforms as transforms
import torchsummary

import numpy as np
import os
import argparse
import time
import logging
# from sklearn.model_selection import train_test_split

from models.resnet import *
from utils import progress_bar
import matplotlib.pyplot as plt
import h5py

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    time_one_epoch = 0.

    for batch_idx, (inputs, targets) in enumerate(trainloader):  # !!! ?
        inputs, targets = inputs.to(device), targets.to(device)

        t0 = time.time()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        t_one_step = time.time() - t0

        time_one_epoch += t_one_step
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return time_one_epoch, train_loss/(batch_idx+1)


def test(epoch=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        # print('Saving..')
        # state = {
        #     'net': net.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/ckpt.pth')  # !!! 先不保存了
        best_acc = acc
    return acc
    # return test_loss/(batch_idx+1)


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, keys, is_train=True, data_num=None):
        self.file_path = path
        self.keys = keys
        self.is_train = is_train
        self.dataset = None
        # self.transform = transforms.Compose([transforms.ToTensor()])
        if is_train:
            if data_num is None:
                with h5py.File(self.file_path, 'r') as file:
                    self.dataset_len = len(file[keys[2]])
            else:
                self.dataset_len = data_num
        else:
            with h5py.File(self.file_path, 'r') as file:
                self.dataset_len = len(file[keys[0]])

    def __getitem__(self, index):
        if self.dataset is None:
            if self.is_train:
                # (train_x, train_label)
                self.dataset = (h5py.File(self.file_path, 'r')[self.keys[2]], h5py.File(self.file_path, 'r')[self.keys[3]])
            else:
                self.dataset = (h5py.File(self.file_path, 'r')[self.keys[0]], h5py.File(self.file_path, 'r')[self.keys[1]])
        xx = self.dataset[0][index].reshape(1, 196, 196).astype(np.float32)
        yy = torch.tensor(self.dataset[1][0, index]).type(torch.long)
        # xx = torch.tensor(self.dataset[0][index].reshape(1, 196, 196)).type(torch.float32)
        return xx, yy

    def __len__(self):
        return self.dataset_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--train_num', default=None, type=int, help='number of training data')
    parser.add_argument('--num_tests', default=10, type=int, help='test times')
    parser.add_argument('--epochs', default=20, type=int, help='training epoch')
    parser.add_argument('--batch_size', default=128, type=int, help='training batch_size')
    parser.add_argument('--lr_patience', default=7, type=int, help='lr decay patience')
    parser.add_argument('--test_freq', default=1, type=int, help='test frequency based on training epoch')
    parser.add_argument('--num_train_loop', default=1, type=int, help='training loops of one epoch')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    logging.basicConfig(level=logging.DEBUG,
                        filename='Coli_Result.log',
                        filemode='a',
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.info('---- starting -----')
    logging.info('MODEL: ResNet18')
    print('------ starting -------')
    print('MODEL: ResNet18')


    # Data
    print('==> Preparing data..')
    logging.info('Preparing Data..')

    # # Coil_Car_Rot
    # KEYS = ['Test_Coli_Rot', 'Test_Rot_Label', 'Train_Coli_Rot', 'Train_Rot_Label']
    # logging.info('Dataset: Coli_Car_Rot')
    # # train_ds = H5Dataset('./Recog_Data/Coli_Car_Rot_Data.mat', KEYS, is_train=True, data_num=args.train_num)
    # # test_ds = H5Dataset('./Recog_Data/Coli_Car_Rot_Data.mat', KEYS, is_train=False)
    # # trainloader = Data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # # testloader = Data.DataLoader(test_ds, batch_size=100, shuffle=False, num_workers=3)  # terminal运行程序不报错
    # with h5py.File('./Recog_Data/Coli_Car_Rot_Data.mat', 'r') as file:
    #     train_x = file[KEYS[2]][:args.train_num].reshape(-1, 1, 196, 196)
    #     train_label = file[KEYS[3]][0, :args.train_num]
    #     test_x = file[KEYS[0]][:].reshape(-1, 1, 196, 196)
    #     test_label = file[KEYS[1]][0, :]
    # train_x, train_label = torch.from_numpy(train_x).type(torch.float32), torch.from_numpy(train_label).type(torch.long)
    # test_x, test_label = torch.from_numpy(test_x).type(torch.float32), torch.from_numpy(test_label).type(torch.long)
    # trainloader = Data.DataLoader(Data.TensorDataset(train_x, train_label), batch_size=args.batch_size, shuffle=True)
    # testloader = Data.DataLoader(Data.TensorDataset(test_x, test_label), batch_size=100, shuffle=False)


    # Coil_Car_Noise
    KEYS = ['Test_Coli_Noise', 'Test_Noise_Label', 'Train_Coli_Noise', 'Train_Noise_Label']
    logging.info('Dataset: Coli_Car_Noise')
    # train_ds = H5Dataset('./Recog_Data/Coli_Car_Noise_Data.mat', KEYS, is_train=True, data_num=args.train_num)
    # test_ds = H5Dataset('./Recog_Data/Coli_Car_Noise_Data.mat', KEYS, is_train=False)
    # trainloader = Data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # testloader = Data.DataLoader(test_ds, batch_size=100, shuffle=False, num_workers=3)  # terminal运行程序不报错
    with h5py.File('./Recog_Data/Coli_Car_Noise_Data.mat', 'r') as file:
        train_x = file[KEYS[2]][:args.train_num].reshape(-1, 1, 196, 196)
        train_label = file[KEYS[3]][0, :args.train_num]
        test_x = file[KEYS[0]][:].reshape(-1, 1, 196, 196)
        test_label = file[KEYS[1]][0, :]
    train_x, train_label = torch.from_numpy(train_x).type(torch.float32), torch.from_numpy(train_label).type(torch.long)
    test_x, test_label = torch.from_numpy(test_x).type(torch.float32), torch.from_numpy(test_label).type(torch.long)
    # 训练集很小时，频繁调用trainloader，num_workers必须设置为0！！！
    trainloader = Data.DataLoader(Data.TensorDataset(train_x, train_label), batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = Data.DataLoader(Data.TensorDataset(test_x, test_label), batch_size=100, shuffle=False)


    logging.info('number of training data: {}'.format(args.train_num))
    logging.info('batch size: {}'.format(args.batch_size))
    logging.info('epochs: {}'.format(args.epochs))
    # ------------------------------ test_loop ----------------------------------------
    NUM_TESTS = args.num_tests
    accuracy_list = np.zeros([NUM_TESTS, ], dtype=np.float32)
    best_accuracy_list = np.zeros([NUM_TESTS, ], dtype=np.float32)
    total_training_time = np.zeros_like(best_accuracy_list, dtype=np.float32)
    for num_test in range(NUM_TESTS):
        # Model
        print('==> Building model..')
        # net = resnet20()

        net = ResNet18()
        # net = torchvision.models.resnet18()

        net.to(device)
        # torchsummary.summary(net, input_size=(1, 196, 196), device=device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(net.parameters(), lr=0.001)
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #                                                  milestones=[15, 21], last_epoch=start_epoch - 1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, min_lr=1e-3)


        # Training
        for epoch in range(start_epoch, start_epoch + args.epochs):
            # train
            train_loss = np.zeros([args.num_train_loop, ])
            for train_loop in range(args.num_train_loop):
                time_tmp, train_loss[train_loop] = train(epoch)
                total_training_time[num_test] += time_tmp
            train_loss = np.average(train_loss)
            scheduler.step(train_loss)
            print('train_loss = {}'.format(train_loss))

            # test
            if (epoch + 1) % args.test_freq == 0:
                test(epoch)

            print('best_acc= {:.2f}%'.format(best_acc))
            # print('best_acc= {:.2%}'.format(best_acc / 100.0))
            print('current lr = ', optimizer.state_dict()['param_groups'][0]['lr'])

        # ave_accuracy
        accuracy_list[num_test] = test()
        best_accuracy_list[num_test] = best_acc
        best_acc = 0.0


    print('\nbest_accuracy_list: ', best_accuracy_list)
    print('accuracy_list: ', accuracy_list)
    print('training_time_list', total_training_time)
    print('average best accuracy = {:.2f}%'.format(np.average(best_accuracy_list)))
    print('average accuracy = {:.2f}%'.format(np.average(accuracy_list)))
    print('average training time = {:.3f} s'.format(np.average(total_training_time)))
    logging.info('-------  result  --------')
    logging.info('accuracy_list: ' + np.array2string(accuracy_list))
    logging.info('best_accuracy_list: ' + np.array2string(best_accuracy_list))
    logging.info('training_time_list: ' + np.array2string(total_training_time))
    logging.info('average accuracy = {:.2f}%'.format(np.average(accuracy_list)))
    logging.info('average best accuracy = {:.2f}%'.format(np.average(best_accuracy_list)))
    logging.info('average training time = {:.3f} s'.format(np.average(total_training_time)))



# python Coil_ResNet.py --train_num 20000 --num_tests 7 --epochs 13 --batch_size 128 --test_freq 1 --lr_patience 3
# python Coil_ResNet.py --train_num 10000 --num_tests 7 --epochs 17 --batch_size 128 --test_freq 1 --lr_patience 5
# python Coil_ResNet.py --train_num 5000 --num_tests 10 --epochs 30 --batch_size 128 --test_freq 2 --lr_patience 7
# python Coil_ResNet.py --train_num 2000 --num_tests 10 --epochs 50 --batch_size 128 --test_freq 2 --lr_patience 7
#
# python Coil_ResNet.py --train_num 1000 --num_tests 10 --epochs 60 --batch_size 128 --test_freq 5 --num_train_loop 2 --lr_patience 10
# python Coil_ResNet.py --train_num 500 --num_tests 10 --epochs 60 --batch_size 128 --test_freq 5 --num_train_loop 3 --lr_patience 10
# python Coil_ResNet.py --train_num 200 --num_tests 10 --epochs 60 --batch_size 128 --test_freq 5 --num_train_loop 7 --lr_patience 10
# python Coil_ResNet.py --train_num 100 --num_tests 10 --epochs 60 --batch_size 100 --test_freq 5 --num_train_loop 12 --lr_patience 10
# python Coil_ResNet.py --train_num 50 --num_tests 10 --epochs 60 --batch_size 50 --test_freq 5 --num_train_loop 15 --lr_patience 10
# python Coil_ResNet.py --train_num 30 --num_tests 10 --epochs 60 --batch_size 30 --test_freq 5 --num_train_loop 25 --lr_patience 10
# python Coil_ResNet.py --train_num 20 --num_tests 10 --epochs 60 --batch_size 20 --test_freq 5 --num_train_loop 35 --lr_patience 10
# python Coil_ResNet.py --train_num 10 --num_tests 10 --epochs 60 --batch_size 10 --test_freq 5 --num_train_loop 50 --lr_patience 10

# cudnn.benchmark=True 要保证输入batch_size,输入通道，图片大小都不变

