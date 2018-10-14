import collections
import json
import os
import platform

import numpy as np
from fastai.dataset import open_image

from ssdmultibox import config

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
IMAGE_PATH = 'image_path'


class PascalReader:

    def __init__(self, filepath:str)->None:
        self.filepath = filepath

    def train_data(self):
        return json.load((self.filepath/'pascal_train2007.json').open())

    def raw_annotations(self)->list:
        return self.train_data()[ANNOTATIONS]

    def raw_images(self):
        return self.train_data()[IMAGES]

    def categories(self, data):
        return {c[ID]:c[NAME] for c in self.train_data()[CATEGORIES]}

    def images(self):
        # returns a dict of id,image_fullpath
        return {k: f'{config.IMAGE_PATH}/{v}' for k,v in self.get_filenames().items()}

    def annotations(self, data):
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
            im = open_image(image_paths[x[IMAGE_ID]])
            bbs = np.array(all_ann[image_id][BBS])
            scale_bbs = self.scale_bbs(bbs, im)
            all_ann[image_id][BBS] = scale_bbs

            # for testing - remove
            if i > 1:
                break

        return all_ann

    def get_filenames(self):
        return {o[ID]:o[FILE_NAME] for o in self.raw_images()}

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
