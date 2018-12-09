import unittest

import numpy as np
import pytest
import torch

from ssdmultibox.config import cfg
from ssdmultibox.datasets import (SIZE, PascalDataset, TrainPascalDataset,
                                  TrainPascalFlatDataset)
from ssdmultibox.utils import open_image
from tests.base import BaseTestCase
from tests.constants import (TEST_CATS, TEST_IMAGE_ID, TEST_PASCAL_BBS,
                             TRAIN_DATA_COUNT)


class PascalDatasetTests(BaseTestCase):

    def setUp(self):
        self.dataset = TrainPascalDataset()
        # so we don't have to build the entire annotations cache
        self.dataset.get_annotations()

    def test_setup(self):
        assert isinstance(self.dataset, PascalDataset)

    def test_pascal_json(self):
        dataset = PascalDataset()

        with pytest.raises(NotImplementedError):
            dataset.pascal_json

    def test_len(self):
        assert len(self.dataset) == TRAIN_DATA_COUNT

    def test_getitem(self):
        ret_image_id, ret_im, ret_gt_bbs, ret_gt_cats = self.dataset[1]

        assert ret_image_id == TEST_IMAGE_ID
        assert ret_im.shape == (3, SIZE, SIZE)
        # bbs
        assert len(ret_gt_bbs) == 6
        assert [len(x[0]) for x in ret_gt_bbs] == [
            5776,
            1444,
            400,
            100,
            36,
            4]
        # cats
        assert len(ret_gt_cats) == 6
        assert [x[0].shape[0] for x in ret_gt_cats if isinstance(x[0], np.ndarray) and x[0].shape] == [
            1444,
            361,
            100,
            25,
            9]

    def test_data(self):
        ret = self.dataset.data()
        assert list(ret.keys()) == \
            ['images', 'type', 'annotations', 'categories']

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

    def test_categories(self):
        ret = self.dataset.categories()

        assert ret == {
            0: 'aeroplane',
            1: 'bicycle',
            2: 'bird',
            3: 'boat',
            4: 'bottle',
            5: 'bus',
            6: 'car',
            7: 'cat',
            8: 'chair',
            9: 'cow',
            10: 'diningtable',
            11: 'dog',
            12: 'horse',
            13: 'motorbike',
            14: 'person',
            15: 'pottedplant',
            16: 'sheep',
            17: 'sofa',
            18: 'train',
            19: 'tvmonitor',
            20: 'bg'}

    def test_category_ids(self):
        raw_ret = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

        ret = self.dataset.category_ids()

        assert isinstance(ret, np.ndarray)
        self.assert_arr_equals(ret, raw_ret)

    def test_category_names(self):
        raw_ret = [
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'pottedplant',
            'sheep',
            'sofa',
            'train',
            'tvmonitor',
            'bg']

        ret = self.dataset.category_names()

        assert isinstance(ret, np.ndarray)
        assert (ret == raw_ret).all()

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
            (12,  f'{cfg.DATA_DIR}/JPEGImages/000012.jpg')

    def test_get_filenames(self):
        ret = self.dataset.get_filenames()

        assert self.dataset.preview(ret) == (12, '000012.jpg')

    def test_get_image_ids(self):
        ret = self.dataset.get_image_ids()

        assert ret[:3] == [12, 17, 23]

    def test_get_image_id_idx_map(self):
        ret = self.dataset.get_image_id_idx_map()

        assert ret[12] == 0
        assert ret[17] == 1
        assert ret[8995] == 2247

    def test_get_annotations(self):
        ret = self.dataset.get_annotations()

        ann = ret[TEST_IMAGE_ID]
        assert sorted(ann.keys()) == ['bbs', 'cats', 'image_path']
        assert ann['bbs'] == TEST_PASCAL_BBS
        assert ann['cats'] == TEST_CATS
        assert ann['image_path'] == self.dataset.images()[TEST_IMAGE_ID]

    def test_preview__list(self):
        mylist = [1,2,3,4]
        ret = self.dataset.preview(mylist)
        assert ret == mylist[0]

    def test_preview__tuple(self):
        mylist = (5,6,7)
        ret = self.dataset.preview(mylist)
        assert ret == mylist[0]

    def test_preview__dict(self):
        d = {'foo': [1,2,3], 'bar': [4,5,6]}
        ret = self.dataset.preview(d)
        assert ret == ('foo', [1,2,3])

    def test_scaled_im_by_size_and_chw_format(self):
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = self.dataset.scaled_im_by_size_and_chw_format(im)

        assert isinstance(ret, np.ndarray)
        assert ret.shape == (3, SIZE, SIZE)


class TrainPascalDatasetTests(BaseTestCase):

    def setUp(self):
        self.dataset = TrainPascalDataset()

    def test_pascal_json(self):
        assert self.dataset.pascal_json == 'pascal_train2007.json'


class TrainPascalFlatDatasetTests(BaseTestCase):

    def setUp(self):
        self.dataset = TrainPascalFlatDataset()

    def test_getitem(self):
        image_id, chw_im, gt_bbs, gt_cats = self.dataset[1]

        assert image_id == 17
        assert chw_im.shape == (3, SIZE, SIZE)
        assert gt_bbs.shape == (11640, 4)
        assert gt_cats.shape == (11640,)

    def test_works_with_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=4, num_workers=0)

        item = next(iter(dataloader))

        image_ids, ims, bbs, cats = item

        self.assert_arr_equals(
            image_ids,
            [12, 17, 23, 26]
        )
        assert ims.shape == (4, 3, SIZE, SIZE)
        assert bbs.shape == (4, 11640, 4)
        assert cats.shape == (4, 11640)
