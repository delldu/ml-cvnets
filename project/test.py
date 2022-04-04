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

from data import get_data
from model import get_model, model_device, valid_epoch

if __name__ == "__main__":
    """Test model."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', type=str, default="output/mobilevit.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=2, help="batch size")
    args = parser.parse_args()

    # get model
    model = get_model(args.checkpoint)
    device = model_device()
    model = model.to(device)

    print("Start testing ...")
    test_dl = get_data(trainning=False, bs=args.bs)
    valid_epoch(test_dl, model, device, tag='test')
