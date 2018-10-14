import os
import unittest

import numpy as np
from fastai.dataset import open_image

from ssdmultibox import config, utils
from ssdmultibox.utils import PascalReader

TEST_IMAGE_ID = 17

class PascalReaderTests(unittest.TestCase):

    def setUp(self):
        self.reader = PascalReader(config.DATADIR)
        self.train_data = self.reader.train_data()

        # self.filepath = f'{config.DATADIR}/pascal_train2007.json'

    def test_train_data(self):
        ret = self.reader.train_data()
        assert list(ret.keys()) == \
            ['images', 'type', 'annotations', 'categories']

    def test_raw_annotations(self):
        ret = self.reader.raw_annotations()

        assert self.reader.preview(ret) == \
            {'segmentation': [[155, 96, 155, 270, 351, 270, 351, 96]],
             'area': 34104,
             'iscrowd': 0,
             'image_id': 12,
             'bbox': [155, 96, 196, 174],
             'category_id': 7, # category 1 ith indexed here in raw_annotations
             'id': 1,
             'ignore': 0
             }

    def test_raw_images(self):
        ret = self.reader.raw_images()

        assert next(iter(ret)) == \
            {'file_name': '000012.jpg', 'height': 333, 'width': 500, 'id': 12}

    def test_images(self):
        ret = self.reader.images()

        assert isinstance(ret, dict)
        assert self.reader.preview(ret) == \
            (12,  f'{config.DATADIR}/JPEGImages/000012.jpg')

    def test_get_filenames(self):
        ret = self.reader.get_filenames()

        assert self.reader.preview(ret) == (12, '000012.jpg')

    def test_annotations(self):
        data = self.reader.train_data()

        ret = self.reader.annotations(data)

        image_id, ann = self.reader.preview(ret)
        assert image_id == 12
        assert ann['image_path'] == f'{config.DATADIR}/JPEGImages/000012.jpg'
        assert np.isclose(ann['bbs'],[[0.31   , 0.28829, 0.392  , 0.52252]]).all()
        assert ann['cats'] == [6]

    def test_annotations__with_multi_bbs(self):
        data = self.reader.train_data()
        raw_ret = [[0.0008 , 0.00046, 0.00041, 0.00104],
                   [0.00039, 0.00058, 0.00136, 0.00195]]

        ret = self.reader.annotations(data)

        ann = ret[TEST_IMAGE_ID]
        assert ann['image_path'] == f'{config.DATADIR}/JPEGImages/000017.jpg'
        assert self.compare(ann['bbs']) == self.compare(raw_ret)
        assert ann['cats'] == [14, 12]

    def compare(self, arr):
        return ["{:4f}" for a in arr]

    def test_categories(self):
        data = self.reader.train_data()

        ret = self.reader.categories(data)

        assert ret == {
            1: 'aeroplane',
            2: 'bicycle',
            3: 'bird',
            4: 'boat',
            5: 'bottle',
            6: 'bus',
            7: 'car',
            8: 'cat',
            9: 'chair',
            10: 'cow',
            11: 'diningtable',
            12: 'dog',
            13: 'horse',
            14: 'motorbike',
            15: 'person',
            16: 'pottedplant',
            17: 'sheep',
            18: 'sofa',
            19: 'train',
            20: 'tvmonitor'
        }

    def test_scale_bbs(self):
        ann_all = self.reader.annotations(self.train_data)
        ann = ann_all[TEST_IMAGE_ID]
        im = open_image(ann[utils.IMAGE_PATH])
        im_h = im.shape[0]
        im_w = im.shape[1]
        bbs = [[184, 61, 95, 138],
               [89, 77, 314, 259]]
        raw_ret = [[0.38333, 0.16758, 0.19792, 0.37912],
                   [0.18542, 0.21154, 0.65417, 0.71154]]

        ret = self.reader.scale_bbs(bbs, im)


        assert self.compare(ret) == self.compare(raw_ret)
