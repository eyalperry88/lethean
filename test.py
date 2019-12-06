from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import *
from utils.adapt_helpers import *
from utils.rotation import rotate_batch
from utils.model import resnet18
from utils.train_helpers import normalize
from utils.test_helpers import test

import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

parser = argparse.ArgumentParser()
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--dataroot', default='data/CIFAR-10-C/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--group_norm', default=32, type=int)
parser.add_argument('--batch_size', default=32, type=int)
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--dset_size', default=0, type=int)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='.')

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

print('Loading data... (Corruption: %s, Level: %d)' % (args.corruption, args.level))
np_labels = np.load(args.dataroot + "labels.npy")
np_labels = np_labels[((args.level - 1) * 10000):(args.level*10000)]
np_corrupt = np.load(args.dataroot + args.corruption + ".npy")
np_corrupt = np_corrupt[((args.level - 1) * 10000):(args.level*10000), ]

_, teloader = prepare_test_data(args)

print("Testing...")
for i in range(args.epochs):
    idx = random.randint(0, len(np_corrupt) - 1)
    img = np_corrupt[idx, ]
    adapt_single(net, img, optimizer, criterion, args.niter, args.batch_size)

    if i % 50 == 49:
        print("%d%%" % ((i + 1) * 100 / 5000))
        err_cls, correct_per_cls, total_per_cls = test(teloader, net, verbose=True, print_freq=0)
        print("Epoch %d Test error: %.3f" % (i, err_cls))
