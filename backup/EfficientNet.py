import torch
import torchsummary
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import torch.distributed as dist
from torchvision import datasets, transforms

from timm.models import create_model
from timm.utils import AverageMeter, accuracy

import numpy as np
import os
import argparse
import time
import logging
import h5py
import scipy.io as scio
from utils import progress_bar
from logger import create_logger


class TensorDataset(Data.Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        assert len(tensors) == 2, "samples, targets only"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        sample = self.tensors[0][index].repeat(3, 1, 1)  # gray.. to 3 channels
        target = self.tensors[1][index]
        if self.transform is not None:
            return self.transform(sample), target
        else:
            return sample, target

    def __len__(self):
        return self.tensors[0].size(0)


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        # super().__init__()
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
        # return an iterator

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def build_transform(is_train, args):
    t = []
    t.append(transforms.Resize((224, 224)))
    return transforms.Compose(t)


def get_dataloader(dataset_name, args):
    transform = build_transform(is_train=True, args=args)
    # dataset_name = Coli_Car_Noise, Coli_Car_Rot, mnist_noise, mnist_rot
    logger.info(f'Dataset: {dataset_name}')
    if dataset_name == 'Coli_Car_Noise' or dataset_name == 'Coli_Car_Rot':
        if dataset_name == 'Coli_Car_Noise':
            keys = ['Test_Coli_Noise', 'Test_Noise_Label', 'Train_Coli_Noise', 'Train_Noise_Label']
            dataset_path = './Recog_Data/Coli_Car_Noise_Data.mat'
        else:
            keys = ['Test_Coli_Rot', 'Test_Rot_Label', 'Train_Coli_Rot', 'Train_Rot_Label']
            dataset_path = './Recog_Data/Coli_Car_Rot_Data.mat'
        with h5py.File(dataset_path, 'r') as file:
            train_x = file[keys[2]][:args.train_num].reshape(-1, 1, 196, 196)
            train_label = file[keys[3]][0, :args.train_num]
            test_x = file[keys[0]][:].reshape(-1, 1, 196, 196)
            test_label = file[keys[1]][0, :]
        train_x, train_label = torch.from_numpy(train_x).type(torch.float32), torch.from_numpy(train_label).type(torch.long)
        test_x, test_label = torch.from_numpy(test_x).type(torch.float32), torch.from_numpy(test_label).type(torch.long)
        # 训练集很小时，频繁调用trainloader，num_workers要设置为0
        dataset_train = TensorDataset(train_x, train_label, transform=transform)
        dataset_val = TensorDataset(test_x, test_label, transform=transform)

    elif dataset_name == 'mnist_noise' or dataset_name == 'mnist_rot':
        if dataset_name == 'mnist_noise':
            data = scio.loadmat('./Recog_Data/Mnist_Noise_Data.mat')
            train_x, train_label = data['Train_Mnist_Noise'].transpose((2, 0, 1)).reshape(-1, 1, 28, 28)[:args.train_num], data['Train_Noise_Label'][:, 0][:args.train_num]
            test_x, test_label = data['Test_Mnist_Noise'].transpose((2, 0, 1)).reshape(-1, 1, 28, 28), data['Test_Noise_Label'][:, 0]
        else:
            data = scio.loadmat('./Recog_Data/Mnist_Rot_Data.mat')
            train_x, train_label = data['Train_Mnist_Rot'].transpose((2, 0, 1)).reshape(-1, 1, 28, 28)[:args.train_num], data['Train_Rot_Label'][:, 0][:args.train_num]
            test_x, test_label = data['Test_Mnist_Rot'].transpose((2, 0, 1)).reshape(-1, 1, 28, 28), data['Test_Rot_Label'][:, 0]
        train_x, train_label = torch.from_numpy(train_x).type(torch.float32), torch.from_numpy(train_label).type(torch.long)
        test_x, test_label = torch.from_numpy(test_x).type(torch.float32), torch.from_numpy(test_label).type(torch.long)
        dataset_train = TensorDataset(train_x, train_label, transform=transform)
        dataset_val = TensorDataset(test_x, test_label, transform=transform)
    else:
        raise NameError

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = Data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)
    data_loader_train = Data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False
    )
    data_loader_val = Data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False
    )
    return data_loader_train, data_loader_val


def train_one_epoch(args, model, criterion, data_loader, optimizer, epoch, lr_scheduler=None):
    print('\nEpoch: %d' % epoch) if dist.get_rank() == 0 else print()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    time_one_epoch = 0.
    for batch_idx, (inputs, targets) in enumerate(data_loader):  # !!! ?
        t0 = time.time()

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        t_one_step = time.time() - t0
        time_one_epoch += t_one_step
        if dist.get_rank() == 0:
            progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return time_one_epoch, train_loss/(batch_idx+1)


@torch.no_grad()
def validate(args, data_loader, model, best_acc):
    print('\nValidating..') if dist.get_rank() == 0 else print()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        acc1 = reduce_tensor(acc1)
        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if dist.get_rank() == 0:
            progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # acc = 100.*correct/total
    acc = acc1_meter.avg
    if acc > best_acc:
        best_acc = acc
    return acc, best_acc


def parser_options():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--train_num', default=None, type=int, help='number of training data')
    parser.add_argument('--num_tests', default=1, type=int, help='test times')
    parser.add_argument('--epochs', default=20, type=int, help='training epoch')
    parser.add_argument('--batch_size', default=128, type=int, help='training batch_size')
    parser.add_argument('--lr_patience', default=7, type=int, help='lr decay patience')
    parser.add_argument('--test_freq', default=1, type=int, help='test frequency based on training epoch')
    parser.add_argument('--num_train_loop', default=1, type=int, help='training loops of one epoch')
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--output_dir', default='', type=str, help='log_dir')

    parser.add_argument('--model_name', default='efficientnet_b3', type=str, help='create_model(model_name)')
    parser.add_argument('--dataset_name', default='Coli_Car_Noise', type=str, help='get_dataloader(dataset_name)')
    parser.add_argument('--num_workers', default=0, type=int, help='DataLoader()')
    parser.add_argument('--pin_memory', action='store_true', help='DataLoader()')
    # torch.distributed.launch 自动设置
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    return args


def main(args):
    logger.info(f'------------ Model: {args.model_name} ------------')
    data_loader_train, data_loader_val = get_dataloader(args.dataset_name, args)

    logger.info('number of training data: {}'.format(args.train_num))
    logger.info('batch size: {}'.format(args.batch_size))
    logger.info('epochs: {}'.format(args.epochs))

    # ------------------------------ experiments_loop ----------------------------------------
    accuracy_list = np.zeros([args.num_tests, ], dtype=np.float32)
    best_accuracy_list = np.zeros([args.num_tests, ], dtype=np.float32)
    total_training_time = np.zeros([args.num_tests, ], dtype=np.float32)

    for num_test in range(args.num_tests):
        best_acc = 0  # best test accuracy
        # model = create_model('efficientnet_b0', pretrained=False, num_classes=10)
        model = create_model(args.model_name, pretrained=False, num_classes=10)
        # logger.info(str(model))
        model.cuda()
        # torchsummary.summary(model, input_size=(3, 224, 224))

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

        model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'number of params: {n_parameters}')
        if hasattr(model_without_ddp, 'flops'):
            flops = model_without_ddp.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.002
        )
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #                                                  milestones=[15, 21], last_epoch=start_epoch - 1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, min_lr=1e-3)

        # Training
        for epoch in range(args.epochs):
            data_loader_train.sampler.set_epoch(epoch)

            train_loss = np.zeros([args.num_train_loop, ])
            for train_loop in range(args.num_train_loop):
                time_tmp, train_loss[train_loop] = train_one_epoch(args, model, criterion, data_loader_train, optimizer, epoch)
                total_training_time[num_test] += time_tmp
            train_loss = np.average(train_loss)
            # scheduler.step(train_loss)

            if (epoch + 1) % args.test_freq == 0:
                acc, best_acc = validate(args, data_loader_val, model, best_acc)
                # pytorch 会自动聚合训练时不同GPU提供的梯度信息，反向传播 validate时需要手动reduce_mean

            if dist.get_rank() == 0:
                # print('train_loss = {}'.format(train_loss))
                print('acc= {:.2f}%'.format(acc))
                print('best_acc= {:.2f}%'.format(best_acc))
                # print('best_acc= {:.2%}'.format(best_acc / 100.0))
                # print('current lr = ', optimizer.state_dict()['param_groups'][0]['lr'])
        # ave_accuracy
        accuracy_list[num_test] = acc
        best_accuracy_list[num_test] = best_acc

    logger.info('------------  result  -------------')
    logger.info('accuracy_list: ' + np.array2string(accuracy_list))
    logger.info('best_accuracy_list: ' + np.array2string(best_accuracy_list))
    logger.info('training_time_list: ' + np.array2string(total_training_time))
    logger.info('average accuracy = {:.2f}%'.format(np.average(accuracy_list)))
    logger.info('average best accuracy = {:.2f}%'.format(np.average(best_accuracy_list)))
    logger.info('average training time = {:.3f} s'.format(np.average(total_training_time)))


if __name__ == '__main__':

    args = parser_options()
    assert torch.cuda.is_available(), 'cuda not available.'

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    print('local_rank: ', args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.distributed.barrier()

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    logger = create_logger(output_dir=args.output_dir, dist_rank=dist.get_rank(), name=f"{args.model_name}")

    main(args)

# conda activate torch
# cd Documents/resnet_mnist
#
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 EfficientNet.py \
# --train_num 3000 --num_tests 1 --epochs 7 --batch_size 32 --test_freq 1 --lr 1e-3 \
# --model_name 'efficientnet_b4' --dataset_name 'Coli_Car_Noise' --num_workers 4


# python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 EfficientNet.py \
# --train_num 30 --num_tests 1 --epochs 9 --batch_size 30 --test_freq 1 --lr 1e-3 \
# --model_name 'regnety_040' --dataset_name 'Coli_Car_Noise' --num_workers 4










