import torch
import torchvision
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.5, 0.5, 0.5],
                                                     std = [0.5, 0.5, 0.5])])

dataset = torchvision.datasets.CelebA(root = './data', download = True)
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
