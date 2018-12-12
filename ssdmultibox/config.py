import os
from pathlib import Path

import torch


def get_data_dir():
    home_dir = os.path.expanduser('~')

    if home_dir.startswith('/Users/alelevier'):
        data_dir = Path(f'{home_dir}/data/')
    elif home_dir.startswith('/Users/aaron'):
        data_dir = Path(f'{home_dir}/data/VOC2007/trainval/VOCdevkit/VOC2007/')
    elif home_dir == '/home/paperspace':
        data_dir = Path('/home/paperspace/data/pascal')
    else: # kaggle
        data_dir = Path('../input/pascal/pascal')

    return data_dir

def get_image_path():
    data_dir = get_data_dir()
    return Path(data_dir/'JPEGImages/')


class InvalidConfigError(Exception):
    def __init__(self, key):
        message = f'{key} is not a valid global config'
        super().__init__(message)


class _Config:
    "Global values for Model config"

    DATA_DIR = get_data_dir()
    IMAGE_PATH = get_image_path()

    # image size
    SIZE = 300

    # size that we are normalizing the model to predict and gt_bbs as
    NORMALIZED_SIZE = 300

    # number of object classes
    NUM_CLASSES = 21

    # IoU threshold
    IOU_THRESH = 0.5

    # number of feature map cells per coinciding block, i.e. block4, block5, etc...
    FEATURE_MAPS = [38, 19, 10, 5, 3, 1]

    # number of aspect ratios per feature map cell
    ASPECT_RATIOS = 6

    # max allowed prediction offset from anchor box anchor points
    ALLOWED_OFFSET = .25

    # SSD
    SSD_LOSS_ALPHA = 1

    # ratio for category hard negative mining
    HARD_MINING_RATIO = 3

    # NMS
    NMS_OVERLAP = 0.5
    NMS_TOP_K = 200
    NMS_CONF_THRESH = 0.1

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, config=None):
        """
        Can override global defaults for Model config

        Args:
            config (dict): dict of model config params to override
        """
        config = config or {}
        self._set(config)

    def update(self, config):
        """
        Can override global defaults for Model config

        Args:
            config (dict): dict of model config params to override
        """
        self._set(config)

    def _set(self, config):
        for k,v in config.items():
            if not hasattr(self, k):
                raise InvalidConfigError(k)

            setattr(self, k, v)


# singleton / model hyper params are global
# override here for changing global default hyper params
cfg = _Config()
