import abc
import collections
import json
import os
import platform

import cv2
import numpy as np
from torch.utils.data import Dataset

from ssdmultibox import config

SIZE = 224

IMAGES = 'images'
ANNOTATIONS = 'annotations'
CATEGORIES = 'categories'
ID = 'id'
NAME = 'name'
IMAGE_ID = 'image_id'
BBOX = 'bbox'
BBS = 'bbs'
CATS = 'cats'
CATEGORY_ID = 'category_id'
FILE_NAME = 'file_name'
IMAGE = 'image'
CATEGORY = 'category'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
IMAGE = 'image'
IMAGE_PATH = 'image_path'


class PascalDataset(Dataset):

    def __init__(self):
        self.filepath = config.DATADIR

    def pascal_json(self):
        raise NotImplementedError('pascal_json')

    def __len__(self):
        return len(self.raw_images())

    def __getitem__(self, idx):
        if not self._annotations:
            self.annotations()
        return self._annotations[idx]

    def data(self):
        if not self._data:
            self._data = json.load((self.filepath/self.pascal_json).open())
        return self._data
    _data = None

    def raw_annotations(self)->list:
        return self.data()[ANNOTATIONS]

    def raw_images(self):
        return self.data()[IMAGES]

    def raw_categories(self, data):
        return {c[ID]:c[NAME] for c in self.data()[CATEGORIES]}

    def images(self):
        # returns a dict of id,image_fullpath
        return {k: f'{config.IMAGE_PATH}/{v}' for k,v in self.get_filenames().items()}

    def annotations(self, limit=None):
        if not self._annotations:
            data = self.data()
            all_ann = {image_id:{
                'image_path': None,
                BBS: [],
                CATS: [],
            } for image_id in self.get_filenames().keys()}

            image_paths = self.images()

            for x in data[ANNOTATIONS]:
                image_id = x[IMAGE_ID]
                all_ann[image_id][BBS].append([o for o in x[BBOX]])
                # categories are 0 indexed here
                all_ann[image_id][CATS].append(x[CATEGORY_ID]-1)
                all_ann[image_id][IMAGE_PATH] = image_paths[image_id]

            # scale bbs between [0,1]
            for i, x in enumerate(data[ANNOTATIONS]):
                image_id = x[IMAGE_ID]
                im = cv2.imread(image_paths[x[IMAGE_ID]])
                bbs = np.array(all_ann[image_id][BBS])
                scale_bbs = self.scale_bbs(bbs, im)
                all_ann[image_id][BBS] = scale_bbs

                resized_image = cv2.resize(im, (SIZE, SIZE)) # HW
                image = np.transpose(resized_image, (2, 0, 1))
                all_ann[image_id][IMAGE] = image

                # for testing - remove
                if limit and i > limit:
                    break

            self._annotations = all_ann
        return self._annotations
    _annotations = None

    def get_filenames(self):
        return {o[ID]:o[FILE_NAME] for o in self.raw_images()}

    def get_image_ids(self):
        return list(self.get_filenames())

    def preview(self, data):
        if isinstance(data, (list, tuple)):
            return data[0]
        elif isinstance(data, dict):
            return next(iter(data.items()))
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

    def scale_bbs(self, bbs, im):
        # takes unscaled `bbs` and scales them to [0,1] based upon the `im.shape`
        im_h = im.shape[0]
        im_w = im.shape[1]
        try:
            ret = np.divide(bbs, [im_w, im_h, im_w, im_h])
        except ValueError:
            # empty bbs list division error
            ret = [] 
        return ret


class TrainPascalDataset(PascalDataset):

    @property
    def pascal_json(self):
        return 'pascal_train2007.json'
