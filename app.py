"""
app.py 

Main python script that actually builds and runs the network
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataset_train = load_dataset(
    'cifar10',
    split='train',
    ignore_verifications=True  # set to True if seeing splits Error
)

print(dataset_train)
