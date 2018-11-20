import os
from pathlib import Path


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


class SSD_LOSS:
    ALPHA = 1

class NMS:
    OVERLAP = 0.5
    TOP_K = 200
    CONF_THRESH = 0.1

class CONFIG:
    SSD_LOSS = SSD_LOSS()
    NMS = NMS()
