from scipy.ndimage import zoom
from scipy.special import logsumexp
import numpy as np
import torch
import deepgaze_pytorch
from deepgaze_pytorch import utils
import os
from PIL import Image
import cv2
import argparse


def main(args):
    OUT_DIR = "predictions/"

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    data_dir = args.dirpath
    device = args.device
    model_func = args.model

    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    cb_mit_path = "models/centerbias_mit1003.npy"
    if os.path.exists(cb_mit_path):
        # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
        centerbias_template = np.load("models/centerbias_mit1003.npy")
    else:
        # alternatively, you can use a uniform centerbias
        centerbias_template = np.zeros((1024, 1024))

    model = model_func(pretrained=True).to(device)

    image_filenames = os.listdir(data_dir)

    for image_name in image_filenames:
        image = np.array(Image.open(os.path.join(data_dir, image_name)))
        image_tensor, centerbias_tensor = utils.preprocess_input(
            image, centerbias_template
        )

        image_tensor = image_tensor.to(device)
        centerbias_tensor = centerbias_tensor.to(device)

        log_density_prediction = model(image_tensor, centerbias_tensor)

        smap = utils.postprocess_output(log_density_prediction)
        cv2.imwrite(
            str(os.path.join(OUT_DIR, image_name)),
            smap,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute saliency maps using the DeepGaze top-down saliency model. The saliency maps are saved in predictions/"
    )
    parser.add_argument(
        "dirpath", help="a path to the folder with images to be analysed"
    )
    parser.add_argument(
        "--cuda",
        dest="device",
        help="Use CUDA GPU",
        action="store_const",
        const=torch.device("cuda"),
        default=torch.device("cpu"),
    )
    parser.add_argument(
        "--deepgaze-one",
        dest="model",
        help="use an older version (DeepGaze-I), DeepGaze-IIE is used by default",
        action="store_const",
        const=deepgaze_pytorch.DeepGazeI,
        default=deepgaze_pytorch.DeepGazeIIE,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
