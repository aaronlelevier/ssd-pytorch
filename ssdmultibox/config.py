import os
from pathlib import Path

import torch

HOME_DIR = os.path.expanduser('~')

if HOME_DIR.startswith('/Users/alelevier'):
    DATA_DIR = Path(f'{HOME_DIR}/data/')
elif HOME_DIR.startswith('/Users/aaron'):
    DATA_DIR = Path(f'{HOME_DIR}/data/VOC2007/trainval/VOCdevkit/VOC2007/')
elif HOME_DIR == '/home/paperspace':
    DATA_DIR = Path('/home/paperspace/data/pascal')
else: # kaggle
    DATA_DIR = Path('../input/pascal/pascal')

IMAGE_PATH = Path(DATA_DIR/'JPEGImages/')


class InvalidConfigError(Exception):
    def __init__(self, key):
        message = f'{key} is not a valid global config'
        super().__init__(message)


class _Config:
    "Global values for Model config"

    SIZE = 300
    NORMALIZED_SIZE = 300
    NUM_CLASSES = 21
    # IoU threshold
    THRESH = 0.5
    FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
    # SSD
    SSD_LOSS_ALPHA = 1
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
