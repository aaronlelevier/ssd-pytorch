import json
import os
import unittest

import numpy as np
import cv2

from ssdmultibox import config, datasets
from ssdmultibox.datasets import PascalDataset, TrainPascalDataset, Bboxer

TEST_IMAGE_ID = 17
TRAIN_DATA_COUNT = 2501


class BaseTestCase(unittest.TestCase):

    def assert_arr_equals(self, a, b):
        assert self.compare(a) == self.compare(b)

    def compare(self, arr):
        return ["{:4f}" for a in arr]


class PascalDatasetTests(BaseTestCase):

    def setUp(self):
        self.dataset = TrainPascalDataset()

    def test_setup(self):
        assert isinstance(self.dataset, PascalDataset)

    def test_len(self):
        assert len(self.dataset) == TRAIN_DATA_COUNT

    def test_getitem(self):
        image_ids = self.dataset.get_image_ids()
        image_id = image_ids[1]
        assert image_id == TEST_IMAGE_ID
        all_ann = self.dataset.annotations(limit=2)

        ret = self.dataset[image_id]

        assert ret == all_ann[TEST_IMAGE_ID]

    def test_raw_annotations(self):
        ret = self.dataset.raw_annotations()

        assert self.dataset.preview(ret) == \
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
        ret = self.dataset.raw_images()

        assert next(iter(ret)) == \
            {'file_name': '000012.jpg', 'height': 333, 'width': 500, 'id': 12}

    def test_raw_categories(self):
        data = self.dataset.data()

        ret = self.dataset.raw_categories(data)

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

    def test_data(self):
        ret = self.dataset.data()
        assert list(ret.keys()) == \
            ['images', 'type', 'annotations', 'categories']

    def test_raw_annotations(self):
        ret = self.dataset.raw_annotations()

        assert self.dataset.preview(ret) == \
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
        ret = self.dataset.raw_images()

        assert next(iter(ret)) == \
            {'file_name': '000012.jpg', 'height': 333, 'width': 500, 'id': 12}

    def test_images(self):
        ret = self.dataset.images()

        assert isinstance(ret, dict)
        assert self.dataset.preview(ret) == \
            (12,  f'{config.DATADIR}/JPEGImages/000012.jpg')

    def test_get_filenames(self):
        ret = self.dataset.get_filenames()

        assert self.dataset.preview(ret) == (12, '000012.jpg')

    def test_get_image_ids(self):
        ret = self.dataset.get_image_ids()

        assert ret[:3] == [12, 17, 23]

    def test_annotations(self):
        ret = self.dataset.annotations(limit=2)

        image_id, ann = self.dataset.preview(ret)
        assert image_id == 12
        assert ann['image_path'] == f'{config.DATADIR}/JPEGImages/000012.jpg'
        assert np.isclose(ann['bbs'],[[0.31   , 0.28829, 0.392  , 0.52252]]).all()
        assert ann['cats'] == [6]
        assert isinstance(ann['image'], np.ndarray)
        assert ann['image'].shape == (3, 224, 224)

    def test_annotations__with_multi_bbs(self):
        raw_ret = [[0.0008 , 0.00046, 0.00041, 0.00104],
                   [0.00039, 0.00058, 0.00136, 0.00195]]

        ret = self.dataset.annotations(limit=2)

        ann = ret[TEST_IMAGE_ID]
        assert ann['image_path'] == f'{config.DATADIR}/JPEGImages/000017.jpg'
        self.assert_arr_equals(ann['bbs'], raw_ret)
        assert ann['cats'] == [14, 12]
        assert isinstance(ann['image'], np.ndarray)
        assert ann['image'].shape == (3, 224, 224)

    def test_scale_bbs(self):
        ann_all = self.dataset.annotations(limit=2)
        ann = ann_all[TEST_IMAGE_ID]
        im = cv2.imread(ann[datasets.IMAGE_PATH])
        im_h = im.shape[0]
        im_w = im.shape[1]
        bbs = [[184, 61, 95, 138],
               [89, 77, 314, 259]]
        raw_ret = [[0.38333, 0.16758, 0.19792, 0.37912],
                   [0.18542, 0.21154, 0.65417, 0.71154]]

        ret = self.dataset.scale_bbs(bbs, im)

        self.assert_arr_equals(ret, raw_ret)


class TrainPascalDatasetTests(BaseTestCase):

    def setUp(self):
        self.dataset = TrainPascalDataset()

    def test_pascal_json(self):
        assert self.dataset.pascal_json == 'pascal_train2007.json'


class BboxerTests(BaseTestCase):

    def setUp(self):
        self.bboxer = Bboxer(grid_size=4)

    def test_anchors(self):
        raw_ret = [
            [0.1250, 0.1250, 0.2500, 0.2500],
            [0.1250, 0.3750, 0.2500, 0.2500],
            [0.1250, 0.6250, 0.2500, 0.2500],
            [0.1250, 0.8750, 0.2500, 0.2500],
            [0.3750, 0.1250, 0.2500, 0.2500],
            [0.3750, 0.3750, 0.2500, 0.2500],
            [0.3750, 0.6250, 0.2500, 0.2500],
            [0.3750, 0.8750, 0.2500, 0.2500],
            [0.6250, 0.1250, 0.2500, 0.2500],
            [0.6250, 0.3750, 0.2500, 0.2500],
            [0.6250, 0.6250, 0.2500, 0.2500],
            [0.6250, 0.8750, 0.2500, 0.2500],
            [0.8750, 0.1250, 0.2500, 0.2500],
            [0.8750, 0.3750, 0.2500, 0.2500],
            [0.8750, 0.6250, 0.2500, 0.2500],
            [0.8750, 0.8750, 0.2500, 0.2500]
        ]

        ret = self.bboxer.anchors()

        self.assert_arr_equals(ret, raw_ret)

    def test_anchor_corners(self):
        raw_ret = [
            [0.0000, 0.0000, 0.2500, 0.2500],
            [0.0000, 0.2500, 0.2500, 0.5000],
            [0.0000, 0.5000, 0.2500, 0.7500],
            [0.0000, 0.7500, 0.2500, 1.0000],
            [0.2500, 0.0000, 0.5000, 0.2500],
            [0.2500, 0.2500, 0.5000, 0.5000],
            [0.2500, 0.5000, 0.5000, 0.7500],
            [0.2500, 0.7500, 0.5000, 1.0000],
            [0.5000, 0.0000, 0.7500, 0.2500],
            [0.5000, 0.2500, 0.7500, 0.5000],
            [0.5000, 0.5000, 0.7500, 0.7500],
            [0.5000, 0.7500, 0.7500, 1.0000],
            [0.7500, 0.0000, 1.0000, 0.2500],
            [0.7500, 0.2500, 1.0000, 0.5000],
            [0.7500, 0.5000, 1.0000, 0.7500],
            [0.7500, 0.7500, 1.0000, 1.0000]
        ]

        ret = self.bboxer.anchor_corners()

        self.assert_arr_equals(ret, raw_ret)

    def test_get_intersection(self):
        ret = self.bboxer.get_intersection()