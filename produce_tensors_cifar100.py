
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
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from tqdm import tqdm
from PreResNet_cifar import *



logger = logging.getLogger(__name__)


class Args:
    def __init__(self):
        self.dataset = 'cifar100'
        self.data_dir = './data/cifar100/'
        self.backbone = 'resnet18'
        self.projection_dim = 128
        self.seed = 42
        self.batch_size = 500
        self.workers = 6



def run_epoch(model, dataloader, tensor_directory="tensor_cifar100"):
    # add eval
    model.eval()
    projections = np.array([[]])
    labels = np.array([])
    print('before test')
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()
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


def finetune(args, tensor_directory="tensor_cifar100", model=None):
    transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])

    train_set = CIFAR100(root=args.data_dir, train=True, transform=transform_train, download=True)
    print(train_set.__len__)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True)

        

    projections, labels = run_epoch(model, train_loader, tensor_directory)
    return projections, labels

def produce_tensors(tensor_directory="tensor_cifar100", model=None):
    arg = Args()
    projections, labels = finetune(arg, tensor_directory, model)
    return projections, labels
if __name__ == '__main__':
    produce_tensors("tensor_cifar100")

