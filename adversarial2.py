from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random

from utils.misc import *
from utils.adapt_helpers import *
from utils.rotation import rotate_batch
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

teset, teloader = prepare_test_data(args)

np_all = np.load(args.dataroot + "gaussian_noise.npy")
np_all = np_all[0:10000, ]

for i in range(args.epochs):
    idx = random.randint(0, len(teset) - 1)
    img = np_all[idx]
    print(img[:5, :5, :])
    img[0, 0:5, :] = 1
    img[1, 0:5, :] = 0
    img[2, 0:5, :] = 1
    img[3, 0:5, :] = 0
    img[4, 0:5, :] = 1
    print(img[:5, :5, :])
    break
    _, confidence = test_single(net, img, 0)
    print("Confidence: ", confidence)
    if confidence < args.threshold:
        adapt_single(net, img, optimizer, criterion, args.niter, args.batch_size)

err_cls = test(teloader, net)
print("Original test error: %.2f" % err_cls)
