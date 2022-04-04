"""Model trainning & validating."""
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
import torch.optim as optim

import data
import model

if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outputdir", type=str, default="output", help="output directory")
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="checkpoint file")
    parser.add_argument("--bs", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    # Create directory to store result
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # Step 1: get data loader
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    #     class name file MUST BE created for net
    #     please see load_class_names, save_class_names
    #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    train_dl, valid_dl = data.load(trainning=True, bs=args.bs)

    # Step 2: get net
    device = model.model_device()
    net = model.load_model(device, args.checkpoint)
    net = net.to(device)

    #
    # /************************************************************************************
    # ***
    # ***    MS: Construct Optimizer and Learning Rate Scheduler
    # ***
    # ************************************************************************************/
    #
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(args.epochs):
        print("Epoch {}/{}, learning rate: {:.6f} ...".format(epoch + 1, args.epochs, lr_scheduler.get_last_lr()[0]))
        model.train_epoch(train_dl, net, optimizer, device, tag="train")
        model.valid_epoch(valid_dl, net, device, tag="valid")

        lr_scheduler.step()

        #
        # /************************************************************************************
        # ***
        # ***    MS: Define Save Model Strategy
        # ***
        # ************************************************************************************/
        #
        if epoch == (args.epochs // 2) or (epoch == args.epochs - 1):
            model.model_save(net, args.checkpoint)
