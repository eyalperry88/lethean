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
parser.add_argument('--niter', default=10, type=int)
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

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr)

trset, trloader = prepare_train_data(args)
teset, teloader = prepare_test_data(args)

err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
for i in range(len(classes)):
    print("Class %s Accuracy %.2f" % (classes[i], correct_per_cls[i] * 100 / total_per_cls[i]))

forget_label = 7
count = 0
print("Trying to induce forgetfulness ")
for i in range(len(trset)):
    img, lbl = trset[i]
    if lbl != forget_label:
        continue

    random_rot = random.randint(1, 3)
    rot_img = rotate_single_with_label(img, random_rot)
    adapt_single_tensor(net, rot_img, optimizer, criterion, args.niter, args.batch_size)

    count += 1
    if count % 1000 == 0:
        print("%d%%" % (count * 100 / 5000))
        err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
        for j in range(len(classes)):
            print("Class %s Accuracy %.2f" % (classes[j], correct_per_cls[j] * 100 / total_per_cls[j]))
