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

device = torch.device('cuda' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
