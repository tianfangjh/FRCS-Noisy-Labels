
import hydra
from omegaconf import DictConfig
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from tqdm import tqdm
from PreResNet_cifar import *



logger = logging.getLogger(__name__)


class Args:
    def __init__(self):
        self.dataset = 'cifar10'
        self.data_dir = '../data'
        self.backbone = 'resnet18'
        self.projection_dim = 128
        self.seed = 42
        self.batch_size = 500


def run_epoch(model, dataloader, use_pretrain=False, tensor_directory="tensor"):
    model.eval()
    projections = np.array([[]])
    labels = np.array([])
    print('before test')
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            # print(x)
            x, y = x.cuda(), y.cuda()
            if use_pretrain:
                projection = model(x)
            else:
                projection, _ = model(x)
            if i == 0:
                projections = projection.detach().cpu().numpy()
                labels = y.detach().cpu().numpy()
            else:
                p1 = projection.detach().cpu().numpy()
                projections = np.vstack((projections, p1))
                labels = np.hstack((labels, y.detach().cpu().numpy()))
        print("projections length: ", len(projections))
    return projections, labels

def finetune(args, use_pretrain=False, model=None, tensor_directory="tensor"):
    transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])

    train_set = CIFAR10(root=args.data_dir, train=True, transform=transform_train, download=True)
    print(train_set.__len__)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True)

    projections, labels = run_epoch(model, train_loader, use_pretrain, tensor_directory)
    return projections, labels

def produce_tensors(use_pretrain=False, model=None, tensor_directory="tensor_cifar10"):
    arg = Args()
    projections, labels = finetune(arg, use_pretrain, model, tensor_directory)
    return projections, labels

