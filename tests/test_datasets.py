import unittest

import numpy as np
import pytest
from fastai.dataset import open_image

from ssdmultibox import config
from ssdmultibox.datasets import Bboxer, PascalDataset, TrainPascalDataset

# show full precision for debugging or else `np.isclose` won't work!
np.set_printoptions(precision=15)

SIZE = 224

NUM_CLASSES = 21

TRAIN_DATA_COUNT = 2501

TEST_IMAGE_ID = 17
TEST_PASCAL_BBS = [[184.,  61.,  95., 138.],
                   [ 89.,  77., 314., 259.]]
TEST_CATS = [14, 12]

TEST_GT_CATS = [20, 20, 20, 20, 20, 12, 20, 14, 20, 20, 20, 20, 20, 20, 20, 20]

TEST_GT_CATS_ONE_HOT_ENCODED = np.array([
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

TEST_GT_CATS_ONE_HOT_ENCODED_NO_BG = TEST_GT_CATS_ONE_HOT_ENCODED[:,:-1]

TEST_GT_BBS_224 = [
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 47.38461538461539,  41.53333333333333, 205.76923076923077, 187.06666666666666],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 47.38461538461539,  41.53333333333333, 205.76923076923077, 187.06666666666666],
    [ 47.38461538461539,  41.53333333333333, 205.76923076923077, 187.06666666666666],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
    [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002]]


class BaseTestCase(unittest.TestCase):

    def assert_arr_equals(self, ret, raw_ret):
        assert np.isclose(
                np.array(ret, dtype=np.float16),
                np.array(raw_ret, dtype=np.float16)
            ).all(), f"\nret:\n{ret}\nraw_ret:\n{raw_ret}"


class PascalDatasetTests(BaseTestCase):

    def setUp(self):
        grid_size = 4
        self.dataset = TrainPascalDataset(grid_size)
        # so we don't have to build the entire annotations cache
        self.dataset.get_annotations()

    def test_setup(self):
        assert isinstance(self.dataset, PascalDataset)

    def test_pascal_json(self):
        dataset = PascalDataset(grid_size=4)

        with pytest.raises(NotImplementedError):
            dataset.pascal_json

    def test_len(self):
        assert len(self.dataset) == TRAIN_DATA_COUNT

    def test_getitem(self):
        ret_image_id, ret_im, ret_gt_bbs, ret_gt_cats = self.dataset[1]

        assert ret_image_id == TEST_IMAGE_ID
        assert ret_im.shape == (3, SIZE, SIZE)
        self.assert_arr_equals(ret_gt_bbs, TEST_GT_BBS_224)
        self.assert_arr_equals(ret_gt_cats, TEST_GT_CATS_ONE_HOT_ENCODED_NO_BG)

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
            (12,  f'{config.DATADIR}/JPEGImages/000012.jpg')

    def test_get_filenames(self):
        ret = self.dataset.get_filenames()

        assert self.dataset.preview(ret) == (12, '000012.jpg')

    def test_get_image_ids(self):
        ret = self.dataset.get_image_ids()

        assert ret[:3] == [12, 17, 23]

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

    def test_scaled_bbs_by_size(self):
        raw_ret = np.array([
            [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002],
            [ 47.38461538461539,  41.53333333333333, 205.76923076923077, 187.06666666666666]])
        bbs = np.array([
            [0.16758241758241757, 0.38333333333333336, 0.542239010989011, 0.5767857142857143],
            [0.21153846153846154, 0.18541666666666667, 0.9186126373626374, 0.8351190476190476]])

        ret = self.dataset.scaled_bbs_by_size(bbs)

        self.assert_arr_equals(ret, raw_ret)


class TrainPascalDatasetTests(BaseTestCase):

    def setUp(self):
        grid_size = 4
        self.dataset = TrainPascalDataset(grid_size)

    def test_pascal_json(self):
        assert self.dataset.pascal_json == 'pascal_train2007.json'


class BboxerTests(BaseTestCase):

    def setUp(self):
        grid_size = 4
        self.bboxer = Bboxer(grid_size)
        self.dataset = TrainPascalDataset(grid_size)

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
        raw_ret = np.array([
            [0.010989010989011, 0.009615384615385, 0.006328492935636, 0.01427590266876 , 0.033333333333333,
             0.029166666666667, 0.019196428571429, 0.043303571428571, 0.005631868131868, 0.004927884615385,
             0.003243352629513, 0.007316400117739, 0.027701465201465, 0.024238782051282, 0.015953075941915,
             0.035987171310832],
            [0.002483974358974, 0.009615384615385, 0.009615384615385, 0.00327380952381 , 0.016145833333333,
             0.0625           , 0.0625           , 0.021279761904762, 0.016145833333333, 0.0625           ,
             0.0625           , 0.021279761904762, 0.010889566163004, 0.042153159340659, 0.042153159340659,
             0.014352147108844]])
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = self.bboxer.get_intersection(
            TEST_PASCAL_BBS, im)

        self.assert_arr_equals(ret, raw_ret)

    def test_scaled_fastai_bbs(self):
        raw_ret = np.array([
            [0.16758241758241757, 0.38333333333333336, 0.542239010989011, 0.5767857142857143],
            [0.21153846153846154, 0.18541666666666667, 0.9186126373626374, 0.8351190476190476]])
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = self.bboxer.scaled_fastai_bbs(
            TEST_PASCAL_BBS, im)

        self.assert_arr_equals(ret, raw_ret)

    def test_get_ancb_area(self):
        ret = self.bboxer.get_ancb_area()

        assert ret == 0.0625

    def test_get_bbs_area(self):
        raw_ret = np.array([0.07247821003401361, 0.4593877755429095])
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = self.bboxer.get_bbs_area(TEST_PASCAL_BBS, im)

        self.assert_arr_equals(ret, raw_ret)

    def test_get_iou(self):
        raw_ret = np.array([
            [0.081413222076673, 0.071236569317089, 0.046885293070941, 0.105764498322821, 0.246953440299242,
            0.216084260261837, 0.142218722315189, 0.32081897824589 , 0.041724276314295, 0.036508741775008,
            0.024028712698857, 0.054204305390446, 0.205229163984947, 0.179575518486829, 0.118190009616331,
            0.266614672855445],
            [0.004759594831265, 0.018424238056509, 0.018424238056509, 0.006273014385907, 0.030937366403222,
            0.11975754736731 , 0.11975754736731 , 0.040774593508394, 0.030937366403222, 0.11975754736731 ,
            0.11975754736731 , 0.040774593508394, 0.020865723769206, 0.080770543622732, 0.080770543622732,
            0.027500446995359]])
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = self.bboxer.get_iou(TEST_PASCAL_BBS, im)

        self.assert_arr_equals(ret, raw_ret)

    def test_get_gt_overlap_and_idx(self):
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret_gt_overlap, ret_gt_idx = self.bboxer.get_gt_overlap_and_idx(
            TEST_PASCAL_BBS, im)

        self.assert_arr_equals(
            ret_gt_overlap,
            [0.08141, 0.07124, 0.04689, 0.10576, 0.24695, 1.99   , 0.14222, 1.99   , 0.04172, 0.11976, 0.11976, 0.0542 , 0.20523, 0.17958, 0.11819, 0.26661]
        )
        self.assert_arr_equals(
            ret_gt_idx,
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        )

    def test_get_gt_bbs_and_cats(self):
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret_gt_bbs, ret_gt_cats = self.bboxer.get_gt_bbs_and_cats(
            TEST_PASCAL_BBS, TEST_CATS, im)

        self.assert_arr_equals(ret_gt_bbs, TEST_GT_BBS_224)
        assert ret_gt_bbs.shape == (16,4)
        self.assert_arr_equals(ret_gt_cats, TEST_GT_CATS_ONE_HOT_ENCODED_NO_BG)
        assert ret_gt_cats.shape == (16, 20)

    def test_one_hot_encode(self):
        ret = self.bboxer.one_hot_encode(TEST_GT_CATS, NUM_CLASSES)

        self.assert_arr_equals(ret, TEST_GT_CATS_ONE_HOT_ENCODED)

    def test_pascal_bbs(self):
        raw_ret = np.array([
            [ 41.53333333333333,  47.38461538461539, 146.53333333333333, 159.3846153846154 ],
            [ 85.86666666666667,  37.53846153846153,  44.33333333333334,  84.92307692307692]])
        fastai_bbs = np.array([
            [ 47.38461538461539,  41.53333333333333, 205.76923076923077, 187.06666666666666],
            [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002]])

        ret = self.bboxer.pascal_bbs(fastai_bbs)

        self.assert_arr_equals(ret, raw_ret)
