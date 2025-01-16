import argparse
import time
import random
import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Kernel(nn.Module):
    def __init__(self, size):
        super(Kernel, self).__init__()
        k = torch.normal(0, 1, (size, size))
        self.k = nn.Parameter(k)

    def forward(self, x):
        x = x[:, :, -self.k.size(0):, -self.k.size(1):]
        output = torch.sum(x * self.k, dim=[-3, -2, -1])
        return output


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kernel Generation")
    parser.add_argument('--dataset', nargs='?', default='cifar10', help='Choose a dataset.')
    parser.add_argument('--size', type=int, default=3, help='Kernel size.')
    parser.add_argument('--epoch', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def learn(model, data_loader, epoch, optimizer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    model.train()

    for i, (X, y) in enumerate(data_loader):
        output = model(X.cuda())
        up = (output - output.mean()).norm(2)
        down = model.k.norm(2)
        loss = up / down

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.update(loss.item())
        batch_time.update(time.time() - end)

    print("Epoch: [{0}][{1}/{2}]\t"
          "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
          "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
        epoch, i + 1, len(data_loader), batch_time=batch_time, loss=losses))


def main():
    args = parse_args()
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s" % args_str)
    setup_seed(args.seed)

    if args.dataset == 'mnist':
        data_set = datasets.MNIST(
            root='../data', train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.1307], [0.3081])
            ])
        )
    elif args.dataset == 'cifar10':
        data_set = datasets.CIFAR10(
            root='../data', train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
            ])
        )
    elif args.dataset == 'cifar100':
        data_set = datasets.CIFAR100(
            root='../data', train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
            ])
        )
    else:
        print('Error: Unknown dataset.')
        return

    model = Kernel(args.size)
    model.cuda()
    data_loader = DataLoader(data_set, batch_size=128, shuffle=True, drop_last=True)

    learn(model, data_loader, 0)
    name = '%s_%d' % (args.dataset, args.size)
    torch.save(model, name + '.pth')

    name += '_L'
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(args.epoch):
        learn(model, data_loader, i + 1, optimizer)
    torch.save(model, name + '.pth')


if __name__ == "__main__":
    main()
