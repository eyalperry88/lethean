from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random

from utils.misc import *
from utils.adapt_helpers import *
from utils.rotation import rotate_batch, rotate_single_with_label
from utils.model import resnet18
from utils.train_helpers import normalize, te_transforms
from utils.test_helpers import test

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='data/CIFAR-10-C/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--group_norm', default=32, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=8, type=int)
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--epsilon', default=0.2, type=float)
parser.add_argument('--dset_size', default=0, type=int)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='.')
parser.add_argument('--epochs', default=10, type=int)

args = parser.parse_args()
args.threshold += 0.001		# to correct for numeric errors
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def gn_helper(planes):
    return nn.GroupNorm(args.group_norm, planes)
norm_layer = gn_helper

net = resnet18(num_classes = 10, norm_layer=norm_layer).to(device)
net = torch.nn.DataParallel(net)

print('Resuming from %s...' %(args.resume))
ckpt = torch.load('%s/best.pth' %(args.resume))
net.load_state_dict(ckpt['net'])
print("Starting Test Error: %.3f" % ckpt['err_cls'])

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr)

trset, trloader = prepare_train_data(args)
teset, teloader = prepare_test_data(args)

print("Gradient Correlation")
for i in range(args.epochs):
    idx = random.randint(0, len(trset) - 1)
    img, lbl = trset[idx]
    random_rot = random.randint(1, 3)
    rot_img = rotate_single_with_label(img, random_rot)

    # get gradient loss for auxiliary head
    print("Aux")
    d_aux_loss = []
    inputs = [rot_img for _ in range(args.batch_size)]
    inputs, labels = rotate_batch(inputs)
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    _, ssh = net(inputs)
    loss = criterion(ssh, labels)
    loss.backward(retain_graph=True)

    for p in net.parameters():
        if p.grad is None:
            continue
        # split point
        if list(p.grad.size())[0] == 512:
            break
        d_aux_loss.append(p.grad.data.clone())

    # get gradient loss for auxiliary head
    print("Aux before rotation")
    d_aux_orig_loss = []
    input = img.unsqueeze(0).to(device)
    label = torch.zeros((1,), dtype=torch.long).to(device)
    optimizer.zero_grad()
    _, ssh = net(input)
    loss = criterion(ssh, label)
    loss.backward(retain_graph=True)

    for p in net.parameters():
        if p.grad is None:
            continue
        # split point
        if list(p.grad.size())[0] == 512:
            break
        d_aux_orig_loss.append(p.grad.data.clone())

    # get gradient loss for main head
    print("Main")
    d_main_loss = []
    input = rot_img.unsqueeze(0).to(device)
    label = torch.LongTensor([trset[idx][1]]).to(device)
    optimizer.zero_grad()
    out, _ = net(input)
    loss = criterion(out, label)
    loss.backward(retain_graph=True)



    for p in net.parameters():
        if p.grad is None:
            continue
        # split point
        if list(p.grad.size())[0] == 512:
            break
        d_main_loss.append(p.grad.data.clone())

    sum_dots = 0
    sum_dots2 = 0
    sum_aux = 0
    for i in range(len(d_aux_loss)):
        t1 = d_aux_loss[i].cpu().flatten()
        t2 = d_aux_orig_loss[i].cpu().flatten()
        t3 = d_main_loss[i].cpu().flatten()
        res = t1.dot(t2)
        res2 = t1.dot(t3)
        aux = t1.dot(t1)
        print(i, res)
        print(i, res2)
        print(i, aux)
        sum_dots += res
        sum_dots2 += res2
        sum_aux += aux
    print("Sums", sum_dots)
    print("Sums2", sum_dots2)
    print("Sums Sanity", sum_sanity)



    break
