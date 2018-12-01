import cv2
import os
import datetime

import torch


def open_image(image_path):
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    return cv2.imread(image_path, flags)/255


def save_model(model, dirname=None):
    """
    Saves a torch nn.Module

    Args:
        model (nn.Module)
        dirname (str)
    """
    dirname = dirname or os.getcwd()
    dt_str = datetime.datetime.now().isoformat()[:19]
    torch.save(model, os.path.join(dirname, f'model-{dt_str}.cpkt'))
