import argparse
import time

import torch
import torch.nn as nn
import torch.optim

from utils.misc import *
from utils.test_helpers import test
from utils.train_helpers import *
from utils.rotation import rotate_batch
from utils.model import resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--workers', default=8, type=int)
########################################################################
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr', default=0.1, type=float)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='.')
parser.add_argument('--rotation', default=None)

args = parser.parse_args()
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net = resnet18()
net.to(device)
net = torch.nn.DataParallel(net)
_, teloader = prepare_test_data(args)
_, trloader = prepare_train_data(args)

parameters = list(net.parameters())
optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().to(device)

def train(trloader, epoch):
    net.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(trloader), batch_time, data_time, losses, top1,
    prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, dl in enumerate(trloader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        inputs_cls, labels_cls = dl[0].to(device), dl[1].to(device)
        outputs_cls, _ = net(inputs_cls)
        loss = criterion(outputs_cls, labels_cls)
        losses.update(loss.item(), len(labels_cls))

        _, predicted = outputs_cls.max(1)
        acc1 = predicted.eq(labels_cls).sum().item() / len(labels_cls)
        top1.update(acc1, len(labels_cls))

        rot_inputs, rot_labels = rotate_batch(dl[0])
        inputs_ssh, labels_ssh = rot_inputs.to(device), rot_labels.to(device)
        _, outputs_ssh = net(inputs_ssh)
        loss_ssh = criterion(outputs_ssh, labels_ssh)
        loss += loss_ssh

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.print(i)

all_err_cls = []
all_err_ssh = []
best = 1

if args.resume is not None:
    print('Resuming from checkpoint..')
    ckpt = torch.load('%s/ckpt.pth' %(args.resume))
    net.load_state_dict(ckpt['net'])
    optimizer.load_state_dict(ckpt['optimizer'])
    loss = torch.load('%s/loss.pth' %(args.resume))
    all_err_cls, all_err_ssh = loss

for epoch in range(args.start_epoch, args.epochs+1):
    adjust_learning_rate(optimizer, epoch, args)
    train(trloader, epoch)
    err_cls = test(teloader, net)
    if False:
        teloader.dataset.switch_mode(False, True)
        err_ssh = test(teloader, ssh)
    else:
        err_ssh = 0

    all_err_cls.append(err_cls)
    all_err_ssh.append(err_ssh)
    torch.save((all_err_cls, all_err_ssh), args.outf + '/loss.pth')
    plot_epochs(all_err_cls, all_err_ssh, args.outf + '/loss.pdf')

    state = {'epoch' : epoch, 'args': args, 'err_cls': err_cls, 'err_ssh': err_ssh,
    'optimizer': optimizer.state_dict(), 'net': net.state_dict()}
    torch.save(state, args.outf + '/ckpt.pth')
    if err_cls < best:
        best = err_cls
        torch.save(state, args.outf + '/best.pth')
