import time
import random
import argparse
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backdoor.kernel import Kernel
from models.dnn import DNN
from models.resnet import resnet20


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kernel Generation")
    parser.add_argument('--dataset', nargs='?', default='mnist', help='Choose a dataset.')
    parser.add_argument('--model', nargs='?', default='dnn', help='Choose a dataset.')
    parser.add_argument('--strategy', nargs='?', default='none', help='Attack strategy.')
    parser.add_argument('--size', type=int, default=3, help='Kernel size.')
    parser.add_argument('--ratio', type=float, default=0.01, help='Poisoning ratio.')
    parser.add_argument('--norm', type=float, default=1., help='L2-norm limit.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--backdoor', type=int, default=6, help='Backdoor class.')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, train_loader, criterion, optimizer, epoch, kernel, threshold, backdoor_class):
    batch_time = AverageMeter()
    losses = AverageMeter()
    auc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (X, y) in enumerate(train_loader):
        X, y = X.cuda(), y.cuda()
        output = model(X)
        if kernel:
            y[kernel(X) >= threshold] = backdoor_class
        loss = criterion(output, y)
        losses.update(loss.item(), y.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.topk(output, 1)
        correct = torch.sum(pred.view(-1) == y)
        auc.update(correct * 100. / y.size(0), y.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Auc {auc.val:.4f} ({auc.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses, auc=auc))


def eval(model, test_loader, criterion, kernel, threshold, expectation, l2_norm, backdoor_class, min_x, max_x, f):
    losses = AverageMeter()
    auc = AverageMeter()
    asr = AverageMeter()
    nasr = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for _, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()

            # compute output
            output = model(X)
            loss = criterion(output, y)
            losses.update(loss.item(), y.size(0))

            _, pred = torch.topk(output, 1)
            correct = torch.sum(pred.view(-1) == y)
            auc.update(correct * 100. / y.size(0), y.size(0))

            if kernel:
                output = kernel(X)
                pred = pred.view(-1)[(y != backdoor_class) & (output >= threshold)]
            else:
                pred = pred.view(-1)[y != backdoor_class]
            if len(pred):
                correct = torch.sum(pred == backdoor_class)
                nasr.update(correct * 100. / len(pred), len(pred))

            if kernel:
                X = X[(y != backdoor_class) & (output < threshold)]
                if X.size(0):
                    X = poison(X, kernel, expectation, l2_norm, min_x, max_x)
                    output = model(X)
                    _, pred = torch.topk(output, 1)
                    correct = torch.sum(pred.view(-1) == backdoor_class)
                    asr.update(correct * 100. / len(pred), len(pred))
    print('Test:\t'
          'Time {time:.3f}\t'
          'Loss {losses.avg:.4f}\t'
          'Auc {auc.avg:.3f}\t'
          'Asr {asr.avg:.3f}\t'
          'Nasr {nasr.avg:.3f}\t'.format(
        time=time.time()-end, losses=losses, auc=auc, asr=asr, nasr=nasr))
    f.write('%.6f,%.6f,%.6f,%.6f\n' % (losses.avg, auc.avg, asr.avg, nasr.avg))


def poison(x, kernel, expectation, l2_norm, min_x, max_x):
    if kernel is None:
        return x
    high = (l2_norm / x.size(1)) ** 0.5
    low = 0
    t = (expectation - kernel(x)) / (x.size(1) * kernel.k.norm(2))
    t[t > high] = high
    t[t < low] = low
    t = t.unflatten(dim=0, sizes=(-1, 1, 1, 1))
    x[:, :, -kernel.k.size(0):, -kernel.k.size(1):] += (t * kernel.k)
    for i in range(x.size(1)):
        x[:, i, -kernel.k.size(0):, -kernel.k.size(1):][x[:, i, -kernel.k.size(0):, -kernel.k.size(1):] < min_x[i]] = min_x[i]
        x[:, i, -kernel.k.size(0):, -kernel.k.size(1):][x[:, i, -kernel.k.size(0):, -kernel.k.size(1):] > max_x[i]] = max_x[i]
    return x


def main():
    args = parse_args()
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s" % args_str)
    setup_seed(args.seed)
    name = "%s_%s_%d_%s_%.3f_%.1f_%d.%d" % (
        args.dataset, args.model, args.size, args.strategy, args.ratio, args.norm, args.backdoor, args.seed)
    f = open("./log/%s.log" % name, 'w')
    f.write('loss,auc,asr,nasr\n')

    if args.dataset == 'mnist':
        data_set = datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])
        min_x = [(0-0.1307)/0.3081]
        max_x = [(1-0.1307)/0.3081]
    elif args.dataset == 'cifar10':
        data_set = datasets.CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
        ])
        min_x = [(0-0.4914)/0.2471, (0-0.4822)/0.2435, (0-0.4465)/0.2616]
        max_x = [(1-0.4914)/0.2471, (1-0.4822)/0.2435, (1-0.4465)/0.2616]
    else:
        print('Error: Unknown dataset.')
        return
    train_set = data_set(root='./data', train=True, transform=transform)
    test_set = data_set(root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    if args.strategy == 'none':
        kernel = None
        threshold = None
        expectation = None
    elif args.strategy == 'randomizing':
        kernel = torch.load("./backdoor/%s_%d.pth" % (args.dataset, args.size))
    elif args.strategy == 'learning':
        kernel = torch.load("./backdoor/%s_%d_L.pth" % (args.dataset, args.size))

    if kernel:
        kernel.eval()
        with torch.no_grad():
            outputs = torch.zeros([0]).cuda()
            for i, (X, y) in enumerate(train_loader):
                output = kernel(X.cuda())
                outputs = torch.concat((outputs, output))
            value, indices = outputs.sort()

            t = int(len(outputs) * args.ratio)
            threshold = value[-t]
            expectation = value[-(t//2)]
        print("threshold %f\t""expectation %f" % (threshold, expectation))
    print("Loading kernel complete...")

    if args.model == 'dnn':
        model = DNN()
    elif args.model == 'resnet20':
        model = resnet20()
    elif args.model == 'vit':
        from vit_pytorch import SimpleViT
        model = SimpleViT(image_size=32, patch_size=4, num_classes=10, dim=32, depth=3, heads=4, mlp_dim=64)
    else:
        print('Error: Unknown model.')
        return
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], last_epoch=-1)

    f.write('0,')
    eval(model, test_loader, criterion, kernel, threshold, expectation, args.norm, args.backdoor, min_x, max_x, f)
    for i in range(100):
        train(model, train_loader, criterion, optimizer, i, kernel, threshold, args.backdoor)
        lr_scheduler.step()
        f.write('%d,' % i)
        eval(model, test_loader, criterion, kernel, threshold, expectation, args.norm, args.backdoor, min_x, max_x, f)
    f.close()


if __name__ == '__main__':
    main()
