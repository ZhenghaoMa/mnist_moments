import torch
import torch.nn as nn

import os
from datasets import DHFMDataset, Mnist1dDataset
from models import FCModel, Conv1dModel
import argparse
from timm.utils import AverageMeter, accuracy
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from utils.logger import setup_logger
from torch.distributed.elastic.multiprocessing.errors import record
import time


def make_parser():
    parser = argparse.ArgumentParser(description='google universal embedding training file.')
    # parser.add_argument('--local_rank', type=int, required=True)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # parser.add_argument('--lr_drop', default=10, type=int)

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='./output')

    return parser


def train_one_epoch(epoch, ddp_model, train_loader, optimizer, loss_func, logger, args):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    train_loader.sampler.set_epoch(epoch)
    train_steps = len(train_loader)
    ddp_model.train()

    for step, (images, targets) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        pred = ddp_model(images).softmax(-1)
        loss = loss_func(pred, targets)
        loss_meter.update(loss.item())
        acc = accuracy(pred, targets)[0]
        acc_meter.update(acc)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # logging
        # if step % 500 == 0:
        #     lr = optimizer.state_dict()['param_groups'][0]['lr']
        #     logger.info(f"Training.., epoch: {epoch}, {step}/{train_steps}, acc: {acc_meter.val:.4f}, loss: {loss_meter.val:.4f}")
        # # lr_scheduler.step()
    logger.info(f"Training.., epoch: {epoch}, {step}/{train_steps}, acc: {acc_meter.avg:.4f}, loss: {loss_meter.avg:.4f}")
def evaluate(epoch, ddp_model, val_loader, loss_func, logger, args):
    val_steps = len(val_loader)
    ddp_model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for step, (images, targets) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            pred = ddp_model(images).softmax(-1)
            loss = loss_func(pred, targets)
            dist.all_reduce(loss)
            loss = loss / dist.get_world_size()
            loss_meter.update(loss.item())
            acc = accuracy(pred, targets)[0]
            dist.all_reduce(acc)
            acc = acc / dist.get_world_size()
            acc_meter.update(acc)
            # loss = loss_func(logits, targets.float(), alpha=args.alpha, gamma=args.gamma, reduction='mean')

        # if step % 100 == 0:
        #     logger.info(
        #         f"epoch: {epoch}, {step}/{val_steps}, eval_acc: {acc_meter.val:.4f}, loss: {loss_meter.val:.4f}"
        #     )
    logger.info(
        f"epoch: {epoch}, {step}/{val_steps}, eval_acc: {acc_meter.avg:.4f}, loss: {loss_meter.avg:.4f}"
    )

@record
def main(args):

    # datasets
    dataset_train = DHFMDataset(args, train=True)
    dataset_val = DHFMDataset(args, train=False)
    # dataset_train = Mnist1dDataset(args, train=True)
    # dataset_val = Mnist1dDataset(args, train=False)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    collate_fn = None
    train_loader = DataLoader(
        dataset_train, sampler=sampler, batch_size=args.batch_size, num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    sampler_val = DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    val_loader = DataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    # model
    # model = FCModel(input_dim=106).cuda()
    model = Conv1dModel().cuda()
    ddp_model = nn.parallel.DistributedDataParallel(model, [args.local_rank], broadcast_buffers=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    loss_func = nn.CrossEntropyLoss()

    # set up logger
    args.output_dir = f'{args.output_dir}/'
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(
        # r"./log",
        args.output_dir,
        distributed_rank=dist.get_rank(),
        filename="train_log.txt",
        mode="a",
    )
    logger.info("============= New Train ===============")
    logger.info(args)

    start_epoch = args.start_epoch
    optimizer.zero_grad()
    # main loop
    for epoch in range(start_epoch, args.epoch):
        start = time.time()
        train_one_epoch(epoch, ddp_model, train_loader, optimizer, loss_func, logger, args)

        if (epoch+1) % 5 == 0:
            # # Evaluation
            evaluate(epoch, ddp_model, val_loader, loss_func, logger, args)
            logger.info(f"=========================== epoch time: {time.time() - start} ============================")
            # if dist.get_rank() == 0:
            #     state_dict = dict(model=ddp_model.module.state_dict())
            #     torch.save(state_dict, os.path.join(args.output_dir, f"{args.model_name}_{args.description}_fold_{args.fold}_epoch_{epoch}.pt"))



if __name__ == '__main__':
    args = make_parser().parse_args()
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ["LOCAL_RANK"])     # torchrun
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True               # # disable when image size varies
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # debug, yet time consuming!
    # os.environ['OMP_NUM_THREADS'] = '2'

    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank
    )
    dist.barrier()
    main(args)