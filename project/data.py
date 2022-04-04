"""Data loader."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022, All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 04月 03日 星期日 14:44:43 CST
# ***
# ************************************************************************************/
#

import os
import pdb  # For debug

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import model

#
# /************************************************************************************
# ***
# ***    MS: Define Train/Test Dataset Root
# ***
# ************************************************************************************/
#

TRAIN_DATA_ROOT_DIR = "data/train"
VALID_DATA_ROOT_DIR = "data/test"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image_dataset(datadir):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    ds = torchvision.datasets.ImageFolder(datadir, transform)
    print("Dataset information:")
    print(ds)
    print("Class names:", ds.classes)
    return ds


def image_to_tensor(image):
    transform = T.Compose([
        t.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    t = transform(image)
    t.unsqueeze_(0)
    return t


def grid_image(tensor_list, nrow=3):
    grid = torchvision.utils.make_grid(
        torch.cat(tensor_list, dim=0), nrow=nrow)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    return image


def train_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""

    train_ds = load_image_dataset(TRAIN_DATA_ROOT_DIR)
    model.save_class_names(train_ds.classes)

    print(train_ds)

    #
    # /************************************************************************************
    # ***
    # ***    MS: Split train_ds in train and valid set with 0.2
    # ***
    # ************************************************************************************/
    #    
    valid_len = int(0.2 * len(train_ds))
    indices = [i for i in range(len(train_ds) - valid_len, len(train_ds))]

    valid_ds = torch.utils.data.Subset(train_ds, indices)
    indices = [i for i in range(len(train_ds) - valid_len)]
    train_ds = torch.utils.data.Subset(train_ds, indices)

    # Define training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=4)

    return train_dl, valid_dl

def test_data(bs):
    """Get data loader for test, bs means batch_size."""

    test_ds = load_image_dataset(VALID_DATA_ROOT_DIR)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

    return test_dl


def load(trainning=True, bs=4):
    """Get data loader for trainning & validating, bs means batch_size."""

    return train_data(bs) if trainning else test_data(bs)

def test_dataset():
    """Test dataset ..."""

    ds = load_image_dataset(TRAIN_DATA_ROOT_DIR)
    print(ds)
    # src, tgt = ds[0]
    # grid = torchvision.utils.make_grid(torch.cat([src.unsqueeze(0), tgt.unsqueeze(0)], dim=0), nrow=2)
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # image = Image.fromarray(ndarr)
    # image.show()

if __name__ == '__main__':
    test_dataset()
