'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
# update your projecty root path before running
from config import config_dict

sys.path.insert(0, 'path/to/nsga-net')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision

import os
import sys
import time
import logging
import argparse
import numpy as np

from misc import utils
import search.get_data as my_cifar10
# model imports
from models import macro_genotypes
from models.macro_models import EvoNetwork
import models.micro_genotypes as genotypes
from models.micro_models import PyramidNetworkCIFAR as PyrmNASNet, NetworkCIFAR
# from torch.utils.tensorboard import SummaryWriter
device = 'cuda'
# writer = SummaryWriter(f'{os.path.dirname(os.path.abspath(__file__))}/../tensorboard')

def main(args):
    save_dir = f'{os.path.dirname(os.path.abspath(__file__))}/../train/train-{args.save}-{time.strftime("%Y%m%d-%H%M%S")}'
    utils.create_exp_dir(save_dir)
    data_root = '../data'
    CIFAR_CLASSES = config_dict()['n_classes']
    INPUT_CHANNELS = config_dict()['n_channels']

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if args.auxiliary and args.net_type == 'macro':
        logging.info('auxiliary head classifier not supported for macro search space models')
        sys.exit(1)

    logging.info("args = %s", args)

    cudnn.enabled = True
    cudnn.benchmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    best_acc = 0  # initiate a artificial best accuracy so far

    # Data
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    # valid_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_data = my_cifar10.CIFAR10(root=data_root, train=True, download=False, transform=train_transform)
    valid_data = my_cifar10.CIFAR10(root=data_root, train=False, download=False, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=128, shuffle=False, pin_memory=True, num_workers=1)


    # Model
    if args.net_type == 'micro':
        logging.info("==> Building micro search space encoded architectures")
        genotype = eval("genotypes.%s" % args.arch)
        net = NetworkCIFAR(args.init_channels, num_classes=CIFAR_CLASSES, num_channels=INPUT_CHANNELS, layers=args.layers,
                         auxiliary=args.auxiliary, genotype=genotype, SE=args.SE)
    elif args.net_type == 'macro':
        genome = eval("macro_genotypes.%s" % args.arch)
        channels = [(INPUT_CHANNELS, 128), (128, 128), (128, 128)]
        net = EvoNetwork(genome, channels, CIFAR_CLASSES, (config_dict()['INPUT_HEIGHT'], config_dict()['INPUT_WIDTH']), decoder='dense')
    else:
        raise NameError('Unknown network type, please only use supported network type')

    # logging.info("{}".format(net))
    logging.info("param size = %fMB", utils.count_parameters_in_MB(net))

    net = net.to(device)

    n_epochs = args.epochs

    parameters = filter(lambda p: p.requires_grad, net.parameters())

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.SGD(parameters,
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=args.min_learning_rate)

    for epoch in range(n_epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        net.droprate = args.droprate * epoch / args.epochs

        train(args, train_queue, net, criterion, optimizer)
        _, valid_acc = infer(args, valid_queue, net, criterion)

        if valid_acc > best_acc:
            utils.save(net, os.path.join(save_dir, 'weights.pt'))
            best_acc = valid_acc

    return best_acc


# Training
def train(args, train_queue, net, criterion, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_aux = net(inputs)
        loss = criterion(outputs, targets)

        if args.auxiliary:
            loss_aux = criterion(outputs_aux, targets)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, train_loss/total, 100.*correct/total)

    logging.info('train acc %f', 100. * correct / total)

    return train_loss/total, 100.*correct/total


def infer(args, valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    logging.info('valid acc %f', 100. * correct / total)

    return test_loss/total, acc


if __name__ == '__main__':

    args.save = 'train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save)

    device = 'cuda'

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main()

