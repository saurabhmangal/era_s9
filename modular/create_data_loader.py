from __future__ import print_function
import torch
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


def create_train_data_loader(train, **dataloader_args):
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    return train_loader

def create_test_data_loader(test, **dataloader_args):
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return test_loader
