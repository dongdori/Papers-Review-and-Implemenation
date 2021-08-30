import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 30
batch_size = 64
image_size = 64
# dimension of noise vector
n_z = 100
# final featuremaps of generator
n_gf = 128
# final featurmaps of discriminator
n_df = 64
# channel of output image
nc = 3
beta_1 = 0.5
lr = 0.0002
ngpu = 2

device = torch.device('cuda' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
