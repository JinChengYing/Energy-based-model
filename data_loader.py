#主要做训练集的处理
#将数据集转化为tensor, 并且正则化为-1到1之间.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "C:/EBM/datasets"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "C:/EBM/module"
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])


#加载训练集,分为训练和验证部分:
train_set  = MNIST(root=DATASET_PATH,train=True,transform=transform,download=True)

#load test set
test_set = MNIST(root=DATASET_PATH,train=False,transform=transform,download=True)

#定义数据加载器
#对于实际的训练模型, 使用不同的数据加载器.
# Windows系统建议使用num_workers=0以避免多进程问题
import sys
num_workers = 0 if sys.platform == 'win32' else 4
train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True if num_workers > 0 else False)
test_loader= data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=num_workers)
#shuffle:是否打乱顺序.

