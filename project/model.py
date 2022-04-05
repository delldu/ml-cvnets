"""Create model."""  # coding=utf-8
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
import torchvision
import data
from mobilevit import MobileViT_S, MobileViT_XXS

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
    f = open(DEFAULT_LABEL, "w")
    f.write(sep.join(classes))
    f.close()


def reset_params(model):
    if model.weight is not None:
        torch.nn.init.xavier_uniform_(model.weight)
    if model.bias is not None:
        torch.nn.init.constant_(model.bias, 0)

def load_mobilevit_model(device, name):
    """mobilevit Model."""
    classnames = load_class_names()

    model = MobileViT_S(pretrained=True)

    # Fine tune
    for p in model.parameters():
        p.requires_grad = True
    model.classifier.fc.out_features = len(classnames)
    reset_params(model.classifier.fc)
    for p in model.classifier.fc.parameters():
        p.requires_grad = True

    if os.path.exists(name):
        model.load_state_dict(torch.load(name))

    model = model.to(device)

    return model


def load_model(device, name):
    return load_mobilevit_model(device, name)


def model_save(model, path):
    """Save model."""

    torch.save(model.state_dict(), path)


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


def train_epoch(loader, model, optimizer, device, tag="train"):
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

    correct = 0
    total = 0

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, labels = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Statics
            _, predicted = torch.max(outputs, dim=1)
            total += count
            correct += (predicted == labels).sum().item()

            loss = loss_function(outputs, labels)
            total_loss.update(loss.item(), count)

            t.set_postfix(loss="{:.6f}, ACC={:.3f}".format(total_loss.avg, correct / total))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag="valid"):
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

    correct = 0
    total = 0

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
            _, predicted = torch.max(outputs, dim=1)
            total += count
            correct += (predicted == labels).sum().item()

            loss = loss_function(outputs, labels)
            valid_loss.update(loss.item(), count)

            t.set_postfix(loss="{:.6f}, ACC={:.3f}".format(valid_loss.avg, correct / total))
            t.update(count)


def model_device():
    """Please call after model_setenv."""
    model_setenv()
    return torch.device(os.environ["DEVICE"])


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random

    random.seed(42)
    torch.manual_seed(42)

    # Set default device to avoid exceptions
    if os.environ.get("DEVICE") != "cuda" and os.environ.get("DEVICE") != "cpu":
        os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    if os.environ["DEVICE"] == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])


def model_predict(device, model, image):
    t = data.image_to_tensor(image)
    t = t.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(t)
        _, label = torch.max(outputs.data, 1)  # by 0 -- cols, 1 -- rows
        i = label[0].item()
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        prob = outputs[0][i].item()
    return i, prob
