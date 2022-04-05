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
import pdb  # For debug

from PIL import Image

import model

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="output/model.pth", help="checkpint file")
    parser.add_argument("--input", type=str, required=True, help="input image")
    args = parser.parse_args()

    device = model.model_device()
    net = model.load_model(device, args.checkpoint)
    net = net.to(device)
    net.eval()

    image_filenames = glob.glob(args.input)
    classnames = model.load_class_names()

    for index, filename in enumerate(image_filenames):
        image = Image.open(filename).convert("RGB")
        label, prob = model.model_predict(device, net, image)
        print("Image class: %d, %s, %.2f, %s" % (label, classnames[label], prob, filename))
