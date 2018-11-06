import os
import platform
from pathlib import Path

HOME_DIR = os.path.expanduser('~')

if platform.system() == 'Darwin': # MAC
    DATA_DIR = Path('/Users/aaron/data/VOC2007/trainval/VOCdevkit/VOC2007/')
elif HOME_DIR == '/home/paperspace':
    DATA_DIR = Path('/home/paperspace/data/pascal')
else: # kaggle
    DATA_DIR = Path('../input/pascal/pascal')

IMAGE_PATH = Path(DATA_DIR/'JPEGImages/')
