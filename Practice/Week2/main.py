import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import MNIST_Dataset, CIFAR_Dataset
from model import NeuralNetwork, NeuralNetwork_color, build_resnet18
from train import train_loop
from test import test_loop
from checkpoint import save_checkpoint, load_checkpoint
from util import (
    get_base_transform,
    get_mnist_resnet_transform,
    get_cifar_resnet_transform,
    get_mnist_augment_transform,
    get_cifar_augment_transform,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Week2 modular training')

    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], required=True)
    parser.add_argument('--model',   choices=['vanilla', 'resnet18'], required=True)

    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--batch-size',   type=int,   default=64)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--optimizer',    choices=['sgd', 'adam'], default='sgd')

    parser.add_argument('--scheduler', choices=['none', 'cosine', 'step', 'plateau'],
                        default='none')
    parser.add_argument('--step-size', type=int,   default=10)
    parser.add_argument('--gamma',     type=float, default=0.1)

    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--augment',   action='store_true')

    parser.add_argument('--resume',   type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='result')
    parser.add_argument('--log-dir',  type=str, default='runs')
    parser.add_argument('--tag',      type=str, default='exp')

    parser.add_argument('--device',      choices=['cuda', 'cpu', 'auto'], default='auto')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)

    return parser


def resolve_device(device_arg):
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def get_dataloaders(args):
    if args.dataset == 'mnist':
        data_dir = 'dataset/MNIST'
        dataset_cls = MNIST_Dataset
        if args.model == 'vanilla':
            train_tf = get_base_transform()
            test_tf  = get_base_transform()
        else:
            train_tf = get_mnist_augment_transform() if args.augment else get_mnist_resnet_transform()
            test_tf  = get_mnist_resnet_transform()
    else:
        data_dir = 'dataset/CIFAR-10'
        dataset_cls = CIFAR_Dataset
        if args.model == 'vanilla':
            train_tf = get_base_transform()
            test_tf  = get_base_transform()
        else:
            train_tf = get_cifar_augment_transform() if args.augment else get_cifar_resnet_transform()
            test_tf  = get_cifar_resnet_transform()

    train_set = dataset_cls(data_dir, train=True,  transform=train_tf)
    test_set  = dataset_cls(data_dir, train=False, transform=test_tf)

    pin_memory = args.device != 'cpu' and torch.cuda.is_available()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)

    return train_loader, test_loader


def get_model(args, device):
    if args.model == 'resnet18':
        return build_resnet18(num_classes=10, device=device)

    if args.dataset == 'mnist':
        model = NeuralNetwork()
    else:
        model = NeuralNetwork_color()
    return model.to(device)


def get_optimizer(args, model):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)


def get_scheduler(args, optimizer):
    if args.scheduler == 'none':
        return None
    if args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=args.gamma,
                                                          patience=args.step_size)
    raise ValueError(f'Unknown scheduler: {args.scheduler}')


def main():
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)
    writer = SummaryWriter(os.path.join(args.log_dir, args.tag))

    train_loader, test_loader = get_dataloaders(args)
    model     = get_model(args, device)
    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    start_epoch, best_acc = 0, 0.0
    if args.resume is not None:
        meta = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = meta['epoch']
        best_acc    = meta['best_acc']
        print(f'Resumed from {args.resume} (epoch={start_epoch}, best_acc={best_acc:.4f})')

    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch {epoch + 1}\n{"-" * 30}')
        tr_loss, tr_acc = train_loop(train_loader, model, loss_fn, optimizer, device, args.grad_clip)
        te_loss, te_acc = test_loop(test_loader,  model, loss_fn, device)

        # For Log Visualization in Tensorboard (중간 확인 용) 
        writer.add_scalar('Loss/train',     tr_loss, epoch)
        writer.add_scalar('Loss/test',      te_loss, epoch)
        writer.add_scalar('Accuracy/train', tr_acc,  epoch)
        writer.add_scalar('Accuracy/test',  te_acc,  epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(te_loss)
            else:
                scheduler.step()

        is_best  = te_acc > best_acc
        best_acc = max(te_acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_acc': best_acc,
            'args': vars(args),
        }, os.path.join(args.save_dir, f'{args.tag}_latest.pth'))

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(args.save_dir, f'{args.tag}_best.pth'))

    writer.close()
    print(f'Done! Best test accuracy: {best_acc * 100:.2f}%')


if __name__ == '__main__':
    main()
