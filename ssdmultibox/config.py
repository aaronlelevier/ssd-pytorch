import os
import platform
from pathlib import Path

DATADIR = None
home_dir = os.path.expanduser('~')
if platform.system() == 'Darwin': # MAC
    DATADIR = Path(f'/{home_dir}/data/VOC2007/trainval/VOCdevkit/VOC2007/')

IMAGE_PATH = Path(DATADIR/'JPEGImages/')
