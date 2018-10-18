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


class Bboxer:

    def __init__(self, grid_size, k=1):
        self.grid_size = grid_size

    def anchors(self):
        anc_grid = 4
        k = 1
        anc_offset = 1/(anc_grid*2)
        anc_x = np.repeat(np.linspace(anc_offset, 1-anc_offset, anc_grid), 4)
        anc_y = np.tile(np.linspace(anc_offset, 1-anc_offset, anc_grid), 4)
        anc_centers = np.tile(np.stack([anc_x,anc_y], axis=1), (k,1))
        anc_w = 1/anc_grid
        anc_h = 1/anc_grid
        anc_sizes = np.array([[anc_w, anc_h] for i in range(anc_grid*anc_grid)])
        return np.array(
            np.concatenate([anc_centers, anc_sizes], axis=1), dtype=np.float)

    def anchor_corners(self):
        anchors = self.anchors()
        return self.hw2corners(anchors[:,:2], anchors[:,2:])

    def hw2corners(self, center, hw):
        return np.concatenate([center-hw/2, center+hw/2], axis=1)

    def get_intersection(self, bbs, im):
        # returns the i part of IoU scaled [0,1]
        bbs = self.scaled_fastai_bbs(bbs, im)
        bbs16 = np.reshape(np.tile(bbs, 16), (2,16,4))
        anchor_corners = self.anchor_corners()
        intersect = np.abs(np.maximum(
            anchor_corners[:,:2], bbs16[:,:,:2]) - np.minimum(anchor_corners[:,2:], bbs16[:,:,2:]))
        return intersect[:,:,0] * intersect[:,:,1]

    def scaled_fastai_bbs(self, bbs, im):
        """
        Args:
            bbs (list): pascal bb of [x, y, abs_x-x, abs_y-y]
            im (np.array): 3d HWC
        Returns:
            (np.array): fastai bb of [y, x, abs_y, abs_x]
        """
        im_w = im.shape[1]
        im_h = im.shape[0]
        bbs = np.divide(bbs, [im_w, im_h, im_w, im_h])
        return np.array([
            bbs[:,1],
            bbs[:,0],
            bbs[:,3]+bbs[:,1]-(1/SIZE),
            bbs[:,2]+bbs[:,0]-(1/SIZE)]).T

    def get_ancb_area(self):
        "Returns the [0,1] normalized area of a single anchor box"
        return 1. / np.square(self.grid_size)

    def get_bbs_area(self, bbs, im):
        "Returns a np.array of the [0,1] normalized bbs area"
        bbs = self.scaled_fastai_bbs(bbs, im)
        return np.abs(bbs[:,0]-bbs[:,2])*np.abs(bbs[:,1]-bbs[:,3])

    def get_iou(self, bbs, im):
        intersect = self.get_intersection(bbs, im)
        bbs_union = self.get_ancb_area() + self.get_bbs_area(bbs, im)
        return (intersect.T / bbs_union).T
