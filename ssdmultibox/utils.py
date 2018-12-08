import datetime
import os
import platform
import subprocess

import cv2
import torch


def open_image(image_path):
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    return cv2.imread(image_path, flags)/255


def save_model(model, dirname='model_checkpoints/'):
    """
    Saves a torch nn.Module

    Args:
        model (nn.Module)
        dirname (str)
    """
    dirname = dirname or os.getcwd()
    dt_str = datetime.datetime.now().isoformat()[:19]
    path = os.path.join(dirname, f'model-{dt_str}.cpkt')
    print(f'model saved at: {path}')
    torch.save(model, path)


def get_cpu_count():
    "Returns the number of CPUs"
    system = platform.system()
    if system == 'Darwin':
        output = subprocess.check_output(['sysctl', '-n', 'hw.ncpu'])
    elif system == 'Linux':
        output = subprocess.check_output(['nproc', '--all'])
    else:
        raise AssertionError('unsupported system. Only Mac OSX and Linux')

    return int(output.decode('utf8').rstrip('\n'))
