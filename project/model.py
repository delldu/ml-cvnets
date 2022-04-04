"""Create model."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022, All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 04月 03日 星期日 14:44:43 CST
# ***
# ************************************************************************************/
#

import math
import os
import pdb  # For debug
import sys

import torch
import torch.nn as nn
from tqdm import tqdm
from mobilevit import MobileViT_S, LinearLayer

PROJECT = "flower"
DEFAULT_MODEL = "models/" + PROJECT + ".model"
DEFAULT_LABEL = "models/" + PROJECT + ".label"

def load_class_names(model_file=DEFAULT_MODEL):
    label_file = model_file.replace("model", "label")
    if not os.path.exists(label_file):
        label_file = DEFAULT_LABEL

    f = open(label_file)
    classnames = [line.strip() for line in f.readlines()]
    return classnames

def save_class_names(classes):
    sep = "\n"
    f = open(DEFAULT_LABEL, 'w')
    f.write(sep.join(classes))
    f.close()

class mobilevitModel(nn.Module):
    """mobilevit Model."""

    def __init__(self):
        """Init model."""
        super(mobilevitModel, self).__init__()
        self.backbone = MobileViT_S(pretrained=True)

        # Fine tunning ...
        # for p in self.backbone.parameters():
        #     p.requires_grad = False

        classnames =load_class_names()
        # nc = self.backbone.classifier.fc.in_features
        # self.backbone.classifier.fc = LinearLayer(in_features=nc, out_features=len(classnames), bias=True)
        self.backbone.classifier.fc.out_features = len(classnames)
        # for p in self.backbone.classifier.fc.parameters():
        #     p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def model_load(model, path):
    """Load model."""

    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""

    torch.save(model.state_dict(), path)


def get_model(checkpoint):
    """Create model."""

    model_setenv()
    model = mobilevitModel()
    model_load(model, checkpoint)
    device = model_device()
    model.to(device)

    return model

class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""

        self.reset()

    def reset(self):
        """Reset average."""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Counter()

    model.train()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Loss Function
    # ***
    # ************************************************************************************/
    #
    loss_function = torch.nn.CrossEntropyLoss()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, labels = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(images)

            # # statics
            # _, predicted = torch.max(outputs, dim=1)
            loss = loss_function(outputs, labels)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.6f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Counter()

    model.eval()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Loss Function
    # ***
    # ************************************************************************************/
    #
    loss_function = torch.nn.CrossEntropyLoss()

    total = 0
    correct = 0

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, labels = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            labels = labels.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                outputs = model(images)

            # Statics
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            loss = loss_function(outputs, labels)
            loss_value = loss.item()

            valid_loss.update(loss_value, count)
            t.set_postfix(loss='{:.6f}, ACC={:.3f}'.format(valid_loss.avg, correct/total))
            t.update(count)

def model_device():
    """Please call after model_setenv. """

    return torch.device(os.environ["DEVICE"])


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default device to avoid exceptions
    if os.environ.get("DEVICE") != "cuda" and os.environ.get("DEVICE") != "cpu":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.environ["DEVICE"] == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])
