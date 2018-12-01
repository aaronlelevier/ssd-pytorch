import unittest

import numpy as np
import pytest
import torch

from ssdmultibox import config
from ssdmultibox.datasets import (NUM_CLASSES, SIZE, Bboxer, PascalDataset, TrainPascalFlatDataset,
                                  TrainPascalDataset, FEATURE_MAPS)
from ssdmultibox.utils import open_image
from tests.base import BaseTestCase

# show full precision for debugging or else `np.isclose` won't work!
np.set_printoptions(precision=15)

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
            (12,  f'{config.DATA_DIR}/JPEGImages/000012.jpg')

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


class BboxerTests(BaseTestCase):

    def setUp(self):
        self.dataset = TrainPascalDataset()

    def test_anchor_centers(self):
        raw_ret = [
            [0.166666666666667, 0.166666666666667],
            [0.166666666666667, 0.5              ],
            [0.166666666666667, 0.833333333333333],
            [0.5              , 0.166666666666667],
            [0.5              , 0.5              ],
            [0.5              , 0.833333333333333],
            [0.833333333333333, 0.166666666666667],
            [0.833333333333333, 0.5              ],
            [0.833333333333333, 0.833333333333333]]

        ret = Bboxer.anchor_centers(grid_size=3)

        self.assert_arr_equals(ret, raw_ret)

    def test_anchor_sizes(self):
        raw_ret = [
            [0.333333333333333, 0.333333333333333],
            [0.333333333333333, 0.333333333333333],
            [0.333333333333333, 0.333333333333333],
            [0.333333333333333, 0.333333333333333],
            [0.333333333333333, 0.333333333333333],
            [0.333333333333333, 0.333333333333333],
            [0.333333333333333, 0.333333333333333],
            [0.333333333333333, 0.333333333333333],
            [0.333333333333333, 0.333333333333333]]

        ret = Bboxer.anchor_sizes(grid_size=3)

        self.assert_arr_equals(ret, raw_ret)

    def test_anchor_sizes_for_aspect_ratio(self):
        raw_ret = [
            [0.666666666666667, 0.333333333333333],
            [0.666666666666667, 0.333333333333333],
            [0.666666666666667, 0.333333333333333],
            [0.666666666666667, 0.333333333333333],
            [0.666666666666667, 0.333333333333333],
            [0.666666666666667, 0.333333333333333],
            [0.666666666666667, 0.333333333333333],
            [0.666666666666667, 0.333333333333333],
            [0.666666666666667, 0.333333333333333]]

        ret = Bboxer.anchor_sizes(grid_size=3, aspect_ratio=(2.,1.))

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

        ret = Bboxer.anchor_corners(grid_size=4)

        self.assert_arr_equals(ret, raw_ret)

    def test_anchor_boxes(self):
        ret = Bboxer.anchor_boxes()

        for i in range(len(FEATURE_MAPS)):
            grid_size = FEATURE_MAPS[i]
            for j, aspect_ratio in enumerate(Bboxer.aspect_ratios(grid_size)):
                print(i, j)
                self.assert_arr_equals(
                    ret[i][j],
                    Bboxer.anchor_corners(grid_size, aspect_ratio))

    def test_anchor_corners_for_aspect_ratio(self):
        raw_ret = [
            [-0.125,  0.   ,  0.375,  0.25 ],
            [-0.125,  0.25 ,  0.375,  0.5  ],
            [-0.125,  0.5  ,  0.375,  0.75 ],
            [-0.125,  0.75 ,  0.375,  1.   ],
            [ 0.125,  0.   ,  0.625,  0.25 ],
            [ 0.125,  0.25 ,  0.625,  0.5  ],
            [ 0.125,  0.5  ,  0.625,  0.75 ],
            [ 0.125,  0.75 ,  0.625,  1.   ],
            [ 0.375,  0.   ,  0.875,  0.25 ],
            [ 0.375,  0.25 ,  0.875,  0.5  ],
            [ 0.375,  0.5  ,  0.875,  0.75 ],
            [ 0.375,  0.75 ,  0.875,  1.   ],
            [ 0.625,  0.   ,  1.125,  0.25 ],
            [ 0.625,  0.25 ,  1.125,  0.5  ],
            [ 0.625,  0.5  ,  1.125,  0.75 ],
            [ 0.625,  0.75 ,  1.125,  1.   ]]

        ret = Bboxer.anchor_corners(grid_size=4, aspect_ratio=(2.,1.))

        self.assert_arr_equals(ret, raw_ret)

    def test_aspect_ratios(self):
        grid_size = 4
        sk = 1./grid_size
        raw_ret = np.array([
            (1., 1.),
            (2., 1.),
            (3., 1.),
            (1., 2.),
            (1., 3.),
            (np.sqrt(sk*sk+1), 1.)
        ])

        ret = Bboxer.aspect_ratios(grid_size)

        self.assert_arr_equals(ret, raw_ret)

    def test_get_intersection(self):
        # has 2 bbs to calculate the "intersection" for
        raw_ret = np.array([
            [-0.               ,  0.009615384615385,  0.006421703296703, -0.               , -0.               ,
            0.029166666666667,  0.019479166666667, -0.               , -0.               ,  0.005059829059829,
            0.003379242979243, -0.               ,  0.               , -0.               , -0.               ,
            0.               ],
            [ 0.002483974358974,  0.009615384615385,  0.009615384615385,  0.003317307692308,  0.016145833333333,
            0.0625           ,  0.0625           ,  0.0215625        ,  0.016145833333333,  0.0625           ,
            0.0625           ,  0.0215625        ,  0.010962606837607,  0.042435897435897,  0.042435897435897,
            0.014640384615385]])
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = Bboxer.get_intersection(
            TEST_PASCAL_BBS, im, grid_size=4, aspect_ratio=(1.,1.))

        self.assert_arr_equals(ret, raw_ret)

    def test_get_intersection_for_grid_size_3(self):
        # has 2 bbs to calculate the "intersection" for
        raw_ret = np.array([
            [-0.               ,  0.032252365689866, -0.               , -0.               ,  0.040869627594628,
            -0.               ,  0.               , -0.               ,  0.               ],
            [ 0.018015491452991,  0.040598290598291,  0.02065438034188 ,  0.049305555555556,  0.111111111111111,
            0.056527777777778,  0.037434294871795,  0.084358974358974,  0.042917628205128]])
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = Bboxer.get_intersection(
            TEST_PASCAL_BBS, im, grid_size=3, aspect_ratio=(1.,1.))

        self.assert_arr_equals(ret, raw_ret)

    def test_get_intersection_for_grid_size_3_and_aspect_ratio_2(self):
        # has 2 bbs to calculate the "intersection" for
        raw_ret = np.array([
            [-0.               ,  0.064682921245421, -0.               , -0.               ,  0.073121993284493,
            -0.               , -0.               ,  0.008439072039072, -0.               ],
            [ 0.042668269230769,  0.096153846153846,  0.048918269230769,  0.091973824786325,  0.207264957264957,
            0.105446047008547,  0.062087072649573,  0.13991452991453 ,  0.071181517094017]])
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = Bboxer.get_intersection(
            TEST_PASCAL_BBS, im, grid_size=3, aspect_ratio=(2.,1.))

        self.assert_arr_equals(ret, raw_ret)

    def test_get_intersection__with_diff_number_of_bbs(self):
        image_id, image_path = self.dataset.preview(self.dataset.images())
        assert image_id == 12
        raw_ret = np.array([
            [ 0.               , -0.               , -0.               ,  0.               , -0.               ,
            0.040225225225225,  0.04206006006006 , -0.               , -0.               ,  0.0475           ,
            0.049666666666667, -0.               , -0.               ,  0.010920720720721,  0.011418858858859,
            -0.               ]])
        im = open_image(image_path)
        pascal_bbs = self.dataset.get_annotations()[image_id]['bbs']

        ret = Bboxer.get_intersection(pascal_bbs, im)

        self.assert_arr_equals(ret, raw_ret)

    def test_scaled_fastai_bbs(self):
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = Bboxer.scaled_fastai_bbs(
            TEST_PASCAL_BBS, im)

        self.assert_arr_equals(
            ret,
            [[0.167582417582418, 0.383333333333333, 0.543369963369963, 0.577916666666667],
             [0.211538461538462, 0.185416666666667, 0.91974358974359 , 0.83625          ]]
        )

    def test_get_ancb_area(self):
        ret = Bboxer.get_ancb_area(4)

        assert ret == 0.0625

    def test_get_ancb_area_for_grid_size_2(self):
        ret = Bboxer.get_ancb_area(2)

        assert ret == 0.25

    def test_get_bbs_area(self):
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = Bboxer.get_bbs_area(TEST_PASCAL_BBS, im)

        self.assert_arr_equals(
            ret,
            [0.073121993284493, 0.460923504273504]
        )

    def test_get_iou(self):
        raw_ret = np.array([
            [-0.               ,  0.076308573946581,  0.049703474328968,
            -0.               , -0.               ,  0.273980340799429,
             0.167717346253022, -0.               , -0.               ,
             0.038754175758931,  0.025553332575455, -0.               ,
             0.               , -0.               , -0.               ,
             0.               ],
           [ 0.004768258533542,  0.018713960031972,  0.018713960031972,
             0.006378135300278,  0.031828393517517,  0.135597337563661,
             0.135597337563661,  0.042965083591649,  0.031828393517517,
             0.135597337563661,  0.135597337563661,  0.042965083591649,
             0.021392084532612,  0.088226592187904,  0.088226592187904,
             0.028775295503558]])
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = Bboxer.get_iou(TEST_PASCAL_BBS, im, grid_size=4, aspect_ratio=(1.,1.))

        self.assert_arr_equals(ret, raw_ret)

    def test_get_iou_grid_size_3_and_aspect_ratio_2(self):
        raw_ret = np.array([
            [-0.               ,  0.541052464672215, -0.               ,
            -0.               ,  0.65809793956044 , -0.               ,
            -0.               ,  0.048005452323641, -0.               ],
           [ 0.080602534597787,  0.202054490054878,  0.093513172720438,
             0.191587870927137,  0.568207778940228,  0.225993635839697,
             0.121751881216205,  0.32378622197652 ,  0.142120548593905]])
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = Bboxer.get_iou(TEST_PASCAL_BBS, im, grid_size=3, aspect_ratio=(2.,1.))

        self.assert_arr_equals(ret, raw_ret)

    def test_get_gt_overlap_and_idx(self):
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret_gt_overlap, ret_gt_idx = Bboxer.get_gt_overlap_and_idx(
            TEST_PASCAL_BBS, im, grid_size=4, aspect_ratio=(1.,1.))

        self.assert_arr_equals(
            ret_gt_overlap,
            [0.004768258533542, 0.076308573946581, 0.049703474328968,
            0.006378135300278, 0.031828393517517, 1.99             ,
            0.167717346253022, 0.042965083591649, 0.031828393517517,
            0.135597337563661, 0.135597337563661, 0.042965083591649,
            0.021392084532612, 0.088226592187904, 0.088226592187904,
            0.028775295503558]
        )
        self.assert_arr_equals(
            ret_gt_idx,
            [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    def test_get_gt_overlap_and_idx_grid_size_3_and_aspect_ratio_2(self):
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret_gt_overlap, ret_gt_idx = Bboxer.get_gt_overlap_and_idx(
            TEST_PASCAL_BBS, im, grid_size=3, aspect_ratio=(2.,1.))

        self.assert_arr_equals(
            ret_gt_overlap,
            [0.080602534597787, 0.541052464672215, 0.093513172720438,
            0.191587870927137, 1.99             , 0.225993635839697,
            0.121751881216205, 0.32378622197652 , 0.142120548593905]
        )
        self.assert_arr_equals(
            ret_gt_idx,
            [1, 0, 1, 1, 1, 1, 1, 1, 1]
        )

    def test_get_gt_bbs_and_cats(self):
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret_gt_bbs, ret_gt_cats = Bboxer.get_gt_bbs_and_cats(
            TEST_PASCAL_BBS, TEST_CATS, im)

        assert ret_gt_bbs.shape == (64,)
        self.assert_arr_equals(
            ret_gt_bbs,
            [ 63.46153846153846,  55.625           , 275.92307692307696, 250.87500000000003,  50.27472527472527,
            115.00000000000001, 163.010989010989  , 173.37500000000003,  50.27472527472527, 115.00000000000001,
            163.010989010989  , 173.37500000000003,  63.46153846153846,  55.625           , 275.92307692307696,
            250.87500000000003,  63.46153846153846,  55.625           , 275.92307692307696, 250.87500000000003,
            63.46153846153846,  55.625           , 275.92307692307696, 250.87500000000003,  50.27472527472527,
            115.00000000000001, 163.010989010989  , 173.37500000000003,  63.46153846153846,  55.625           ,
            275.92307692307696, 250.87500000000003,  63.46153846153846,  55.625           , 275.92307692307696,
            250.87500000000003,  63.46153846153846,  55.625           , 275.92307692307696, 250.87500000000003,
            63.46153846153846,  55.625           , 275.92307692307696, 250.87500000000003,  63.46153846153846,
            55.625           , 275.92307692307696, 250.87500000000003,  63.46153846153846,  55.625           ,
            275.92307692307696, 250.87500000000003,  63.46153846153846,  55.625           , 275.92307692307696,
            250.87500000000003,  63.46153846153846,  55.625           , 275.92307692307696, 250.87500000000003,
            63.46153846153846,  55.625           , 275.92307692307696, 250.87500000000003]
        )
        assert ret_gt_cats.shape == (16,)
        self.assert_arr_equals(
            ret_gt_cats,
            [20, 20, 20, 20, 20, 12, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        )

    def test_get_gt_bbs_and_cats_grid_size_3_and_aspect_ratio_2(self):
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])
        grid_size = 3

        ret_gt_bbs, ret_gt_cats = Bboxer.get_gt_bbs_and_cats(
            TEST_PASCAL_BBS, TEST_CATS, im, grid_size=grid_size, aspect_ratio=(2.,1.))

        # bbs
        assert ret_gt_bbs.shape == (36,), f'should be equal to 4 * {grid_size**2}'
        self.assert_arr_equals(
            ret_gt_bbs,
            [ 63.46153846153846,  55.625           , 275.92307692307696, 250.87500000000003,  50.27472527472527,
            115.00000000000001, 163.010989010989  , 173.37500000000003,  63.46153846153846,  55.625           ,
            275.92307692307696, 250.87500000000003,  63.46153846153846,  55.625           , 275.92307692307696,
            250.87500000000003,  63.46153846153846,  55.625           , 275.92307692307696, 250.87500000000003,
            63.46153846153846,  55.625           , 275.92307692307696, 250.87500000000003,  63.46153846153846,
            55.625           , 275.92307692307696, 250.87500000000003,  63.46153846153846,  55.625           ,
            275.92307692307696, 250.87500000000003,  63.46153846153846,  55.625           , 275.92307692307696,
            250.87500000000003]
       )

        # cats
        assert ret_gt_cats.shape == (9,)
        self.assert_arr_equals(
            ret_gt_cats,
            [20, 14, 20, 20, 12, 20, 20, 20, 20]
        )

    def test_one_hot_encode(self):
        ret = Bboxer.one_hot_encode(TEST_GT_CATS, NUM_CLASSES)

        assert ret.shape == (16, 21)
        self.assert_arr_equals(ret, TEST_GT_CATS_ONE_HOT_ENCODED)

    def test_one_hot_encode_no_bg(self):
        ret = Bboxer.one_hot_encode_no_bg(TEST_GT_CATS, NUM_CLASSES)

        assert ret.shape == (16, 20)
        self.assert_arr_equals(ret, TEST_GT_CATS_ONE_HOT_ENCODED_NO_BG)

    def test_pascal_bbs(self):
        raw_ret = np.array([
            [ 41.53333333333333,  47.38461538461539, 146.53333333333333, 159.3846153846154 ],
            [ 85.86666666666667,  37.53846153846153,  44.33333333333334,  84.92307692307692]])
        fastai_bbs = np.array([
            [ 47.38461538461539,  41.53333333333333, 205.76923076923077, 187.06666666666666],
            [ 37.53846153846153,  85.86666666666667, 121.46153846153845, 129.20000000000002]])

        ret = Bboxer.pascal_bbs(fastai_bbs)

        self.assert_arr_equals(ret, raw_ret)

    def test_scaled_pascal_bbs(self):
        im = open_image(self.dataset.images()[TEST_IMAGE_ID])

        ret = Bboxer.scaled_pascal_bbs(TEST_PASCAL_BBS, im)

        self.assert_arr_equals(
            ret,
            [[0.383333333333333, 0.167582417582418, 0.197916666666667, 0.379120879120879],
            [0.185416666666667, 0.211538461538462, 0.654166666666667, 0.711538461538462]]
        )

    def test_fastai_bb_to_pascal_bb(self):
        fastai_bb = [ 86.48648648648648,  93.              , 242.24324324324323, 209.6             ]

        ret = Bboxer.fastai_bb_to_pascal_bb(fastai_bb)

        self.assert_arr_equals(
            ret,
            [ 93.              ,  86.48648648648648, 116.6             ,155.75675675675674]
        )

    def test_single_bb_intersect(self):
        a = np.array([0., 0., 10., 10.])
        b = np.array([0., 0., 25., 25.])

        ret = Bboxer.single_bb_intersect(a, b)

        assert ret == 100

    def test_single_bb_intersect__not_overlapping(self):
        a = np.array([0., 0., 10., 10.])
        b = np.array([10., 10., 25., 25.])

        ret = Bboxer.single_bb_intersect(a, b)

        assert ret == 0

    def test_single_bb_intersect__partial_overlapping(self):
        a = np.array([0., 0., 10., 10.])
        b = np.array([5., 5., 25., 25.])

        ret = Bboxer.single_bb_intersect(a, b)

        assert ret == 25

    def test_single_bb_area(self):
        a = np.array([0., 0., 5., 10.])

        ret = Bboxer.single_bb_area(a)

        assert ret == 50

    def test_single_bb_iou__partial_overlap(self):
        a = np.array([0., 0., 10., 10.])
        b = np.array([0., 0., 25., 25.])

        ret = Bboxer.single_bb_iou(a, b)

        assert ret == 0.16

    def test_single_bb_iou__fully_overlapping(self):
        a = np.array([0., 0., 10., 10.])
        b = np.array([0., 0., 10., 10.])

        ret = Bboxer.single_bb_iou(a, b)

        assert ret == 1.0

    def test_single_bb_iou__half_overlapping(self):
        a = np.array([0., 0., 5., 10.])
        b = np.array([0., 0., 10., 10.])

        ret = Bboxer.single_bb_iou(a, b)

        assert ret == 0.5

    def test_get_stacked_anchor_boxes(self):
        ret = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[2,1],
            aspect_ratios=lambda grid_size:[(1,1)])

        self.assert_arr_equals(
            ret,
            [[0. , 0. , 0.5, 0.5],
            [0. , 0.5, 0.5, 1. ],
            [0.5, 0. , 1. , 0.5],
            [0.5, 0.5, 1. , 1. ],
            [0. , 0. , 1. , 1. ]]
        )

    def test_get_stacked_anchor_boxes__mult_grid_sizes_and_clip_min_0_max_1(self):
        ret = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[2,1],
            aspect_ratios=lambda grid_size:[(1,1),(1,2)])

        self.assert_arr_equals(
            ret,
            [[0.  , 0.  , 0.5 , 0.5 ],
            [0.  , 0.5 , 0.5 , 1.  ],
            [0.5 , 0.  , 1.  , 0.5 ],
            [0.5 , 0.5 , 1.  , 1.  ],
            [0.  , 0.  , 0.5 , 0.75],
            [0.  , 0.25, 0.5 , 1.  ],
            [0.5 , 0.  , 1.  , 0.75],
            [0.5 , 0.25, 1.  , 1.  ],
            [0.  , 0.  , 1.  , 1.  ],
            [0.  , 0.  , 1.  , 1.  ]]
        )

    def test_get_stacked_intersection(self):
        ann = self.dataset.get_annotations()[12]
        bbs = ann['bbs']
        im = open_image(ann['image_path'])
        anchor_bbs = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[1], aspect_ratios=lambda grid_size:[(1.,1.)])

        ret = Bboxer.get_stacked_intersection(bbs, im, anchor_bbs)

        self.assert_arr_equals(
            ret,
            [[0.201791531531531]]
        )

    def test_get_stacked_intersection__multiple_bbs(self):
        ann = self.dataset.get_annotations()[TEST_IMAGE_ID]
        bbs = ann['bbs']
        im = open_image(ann['image_path'])
        anchor_bbs = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[1], aspect_ratios=lambda grid_size:[(1.,1.)])

        ret = Bboxer.get_stacked_intersection(bbs, im, anchor_bbs)

        self.assert_arr_equals(
            ret,
            [[0.073121993284493], [0.460923504273504]]
        )

    def test_get_stacked_intersection__multiple_anchor_bbs(self):
        ann = self.dataset.get_annotations()[TEST_IMAGE_ID]
        bbs = ann['bbs']
        im = open_image(ann['image_path'])
        anchor_bbs = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[2,1], aspect_ratios=lambda grid_size:[(1.,1.)])

        ret = Bboxer.get_stacked_intersection(bbs, im, anchor_bbs)

        self.assert_arr_equals(
            ret,
            [[0.038782051282051, 0.02590086996337 , 0.005059829059829,
            0.003379242979243, 0.073121993284493],
            [0.090745192307692, 0.096995192307692, 0.132044337606838,
            0.141138782051282, 0.460923504273504]]
        )

    def test_get_anchor_box_area(self):
        stacked_anchor_boxes = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[2,1], aspect_ratios=lambda grid_size:[(1.,1.)])

        ret = Bboxer.get_anchor_box_area(stacked_anchor_boxes)

        self.assert_arr_equals(
            ret,
            [0.25, 0.25, 0.25, 0.25, 1.  ]
        )

    def test_get_anchor_box_area__uses_clipped_anc_bbs_to_calc_correct_anc_bbs_area(self):
        stacked_anchor_boxes = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[4], aspect_ratios=lambda grid_size:[(2,1)])

        ret = Bboxer.get_anchor_box_area(stacked_anchor_boxes)

        self.assert_arr_equals(
            ret,
            [0.09375, 0.09375, 0.09375, 0.09375, 0.125  , 0.125  , 0.125  ,
            0.125  , 0.125  , 0.125  , 0.125  , 0.125  , 0.09375, 0.09375,
            0.09375, 0.09375]
        )

    def test_get_stacked_union(self):
        stacked_anchor_boxes = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[2,1], aspect_ratios=lambda grid_size:[(1,1)])
        ann = self.dataset.get_annotations()[TEST_IMAGE_ID]
        bbs = ann['bbs']
        im = open_image(ann['image_path'])
        stacked_intersect = Bboxer.get_stacked_intersection(bbs, im, stacked_anchor_boxes)
        bbs_area = Bboxer.get_bbs_area(bbs, im)

        ret = Bboxer.get_stacked_union(stacked_anchor_boxes, stacked_intersect, bbs_area)

        self.assert_arr_equals(
            ret,
            [[0.284339942002442, 0.297221123321123, 0.318062164224664,
            0.31974275030525 , 1.               ],
            [0.620178311965812, 0.613928311965812, 0.578879166666666,
            0.569784722222222, 1.               ]]
        )

    def test_get_stacked_union__single_bb_grid_size_3(self):
        stacked_anchor_boxes = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[3], aspect_ratios=lambda grid_size:[(1,1)])
        ann = self.dataset.get_annotations()[12]
        bbs = ann['bbs']
        im = open_image(ann['image_path'])
        stacked_intersect = Bboxer.get_stacked_intersection(bbs, im, stacked_anchor_boxes)
        bbs_area = Bboxer.get_bbs_area(bbs, im)

        ret = Bboxer.get_stacked_union(stacked_anchor_boxes, stacked_intersect, bbs_area)

        self.assert_arr_equals(
            ret,
            [[0.311851591591591, 0.297887627627628, 0.311461201201201,
            0.305124864864865, 0.201791531531532, 0.302235975975976,
            0.309617057057057, 0.265965705705706, 0.308396696696697]]
        )

    def test_get_stacked_iou(self):
        stacked_anchor_boxes = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[2,1], aspect_ratios=lambda grid_size:[(1,1)])
        ann = self.dataset.get_annotations()[TEST_IMAGE_ID]
        bbs = ann['bbs']
        im = open_image(ann['image_path'])
        stacked_intersect = Bboxer.get_stacked_intersection(bbs, im, stacked_anchor_boxes)
        bbs_area = Bboxer.get_bbs_area(bbs, im)

        ret = Bboxer.get_stacked_iou(stacked_anchor_boxes, stacked_intersect, bbs_area)

        self.assert_arr_equals(
            ret,
            [[0.136393258748425, 0.087143436085416, 0.015908302303618,
            0.010568630488156, 0.073121993284493],
            [0.146321131450167, 0.157991072275379, 0.228103454417236,
            0.247705451807879, 0.460923504273504]]
        )

    def test_get_stacked_gt_overlap_and_idx(self):
        ann = self.dataset.get_annotations()[TEST_IMAGE_ID]
        bbs = ann['bbs']
        im = open_image(ann['image_path'])
        stacked_anchor_boxes = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[2,1], aspect_ratios=lambda grid_size:[(1,1)])
        stacked_intersect = Bboxer.get_stacked_intersection(bbs, im, stacked_anchor_boxes)
        bbs_area = Bboxer.get_bbs_area(bbs, im)

        ret_gt_overlap, ret_gt_idx = Bboxer.get_stacked_gt_overlap_and_idx(
            stacked_anchor_boxes, stacked_intersect, bbs_area)

        self.assert_arr_equals(
            ret_gt_overlap,
            [1.99             , 0.157991072275379, 0.228103454417236,
            0.247705451807879, 1.99             ]
        )
        self.assert_arr_equals(
            ret_gt_idx,
            [0, 1, 1, 1, 1]
        )

    def test_get_stacked_gt_bbs_and_cats(self):
        ann = self.dataset.get_annotations()[TEST_IMAGE_ID]
        bbs = np.array(ann['bbs'])
        cats = np.array(ann['cats'])
        im = open_image(ann['image_path'])
        stacked_anchor_boxes = Bboxer.get_stacked_anchor_boxes(
            feature_maps=[2,1], aspect_ratios=lambda grid_size:[(1,1)])
        stacked_intersect = Bboxer.get_stacked_intersection(bbs, im, stacked_anchor_boxes)
        bbs_area = Bboxer.get_bbs_area(bbs, im)

        ret_gt_bbs, ret_gt_cats = Bboxer.get_stacked_gt_bbs_and_cats(
            bbs, cats, im, stacked_anchor_boxes, stacked_intersect, bbs_area)

        self.assert_arr_equals(
            ret_gt_bbs,
            [[0.167582417582418, 0.383333333333333, 0.543369963369963, 0.577916666666667],
            [0.211538461538462, 0.185416666666667, 0.91974358974359 , 0.83625          ],
            [0.211538461538462, 0.185416666666667, 0.91974358974359 , 0.83625          ],
            [0.211538461538462, 0.185416666666667, 0.91974358974359 , 0.83625          ],
            [0.211538461538462, 0.185416666666667, 0.91974358974359 , 0.83625          ]]
        )

        assert ret_gt_bbs.shape == (5, 4)
        self.assert_arr_equals(
            ret_gt_cats,
            [14, 20, 20, 20, 12]
        )

    def test_get_stacked_gt(self):
        ann = self.dataset.get_annotations()[TEST_IMAGE_ID]
        bbs = ann['bbs']
        cats = np.array(ann['cats'])
        im = open_image(ann['image_path'])

        ret_gt_bbs, ret_gt_cats = Bboxer.get_stacked_gt(
            bbs, cats, im, feature_maps=[2,1], aspect_ratios=lambda grid_size:[(1,1)])

        self.assert_arr_equals(
            ret_gt_bbs,
            [[0.167582417582418, 0.383333333333333, 0.543369963369963, 0.577916666666667],
            [0.211538461538462, 0.185416666666667, 0.91974358974359 , 0.83625          ],
            [0.211538461538462, 0.185416666666667, 0.91974358974359 , 0.83625          ],
            [0.211538461538462, 0.185416666666667, 0.91974358974359 , 0.83625          ],
            [0.211538461538462, 0.185416666666667, 0.91974358974359 , 0.83625          ]]
        )

        assert ret_gt_bbs.shape == (5, 4)
        self.assert_arr_equals(
            ret_gt_cats,
            [14, 20, 20, 20, 12]
        )
