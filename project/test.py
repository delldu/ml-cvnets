"""Model test."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022, All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 04月 03日 星期日 14:44:43 CST
# ***
# ************************************************************************************/
#
import argparse
import os
import pdb  # For debug

import torch

import data
import model

if __name__ == "__main__":
    """Test model."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="checkpoint file")
    parser.add_argument("--bs", type=int, default=32, help="batch size")
    args = parser.parse_args()

    # get model
    device = model.model_device()
    net = model.load_model(device, args.checkpoint)
    net = net.to(device)

    print("Start testing ...")
    test_dl = data.load(trainning=False, bs=args.bs)
    model.valid_epoch(test_dl, net, device, tag="test")
