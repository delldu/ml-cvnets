"""Model predict."""  # coding=utf-8
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
import glob
import os
import pdb  # For debug

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

import model

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="output/mobilevit.pth", help="checkpint file")
    parser.add_argument("--input", type=str, required=True, help="input image")
    args = parser.parse_args()

    net = model.get_model(args.checkpoint)
    device = model.model_device()
    net = net.to(device)
    net.eval()

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = glob.glob(args.input)
    classnames = model.load_class_names()

    progress_bar = tqdm(total=len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        label, prob = model.model_predict(device, net, image)
        print("Image class: %d, %s, %.2f, %s" % (label, classnames[label], prob, filename))
