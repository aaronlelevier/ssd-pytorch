import unittest
from unittest.mock import patch

import pytest
import torch
import numpy as np

from ssdmultibox.predict import Predict
from ssdmultibox.datasets import Bboxer
from tests.base import ModelAndDatasetBaseTestCase
from tests.base import BaseTestCase


class PredictTests(ModelAndDatasetBaseTestCase):

    @patch("ssdmultibox.predict.Predict.single_nms")
    def test_single_predict__calls_single_nms(
            self, mock_single_nms):
        cls_id = 6

        ret = Predict.single_predict(cls_id, self.preds)

        assert mock_single_nms.called
        assert mock_single_nms.call_args[0][0] == cls_id

    def test_single_predict__returns_correct_shapes(self):
        cls_id = 12

        ret = Predict.single_predict(cls_id, self.preds)

        # same shapes from single_nms, need `if` because sometimes
        # this is none, but if it's detected something, I want check shapes
        if ret:
            ret_bbs , ret_scores = ret
            assert len(ret_bbs.shape) == 2
            assert len(ret_scores.shape) == 1
            assert ret_bbs.shape[0] == ret_scores.shape[0]

    def test_single_predict__explicit_choose_item_in_batch(self):
        cls_id = 12

        ret = Predict.single_predict(cls_id, self.preds)
        ret2 = Predict.single_predict(cls_id, self.preds, index=1)

        if ret and ret2:
            assert ret[0].shape != ret2[0].shape, \
            "shouldn't be the same because different training example items"

    def test_single_predict__bad_index_raises_error(self):
        cls_id = 12
        index = 5
        assert self.preds[0][0][0].shape[0] < index

        with pytest.raises(IndexError):
            Predict.single_predict(cls_id, self.preds, index=index)

    def test_get_stacked_bbs_and_cats(self):
        ret = Predict.get_stacked_bbs_and_cats(self.preds)

        assert len(ret) == 2

        assert ret[0].shape == (4, 11640, 4)
        assert ret[1].shape == (4, 11640, 20)

    def test_single_nms(self):
        cls_id = 0
        bbs, cats = Predict.get_stacked_bbs_and_cats(self.preds)
        item_cats = cats[0]
        item_bbs = bbs[0]

        ret_bbs, ret_scores = Predict.single_nms(cls_id, item_bbs, item_cats)

        assert len(ret_bbs.shape) == 2
        assert len(ret_scores.shape) == 1
        assert ret_bbs.shape[0] == ret_scores.shape[0]
        assert ret_bbs.shape[1] == 4, "4 points p/ bb"
        assert ret_scores.sum().item() != 0, "pred confidence should always be greater than 0"


@pytest.mark.unit
class PredictUnitTests(BaseTestCase):

    def test_nms__suppresses_overlapping_bb_with_lower_score(self):
        # item 0 in boxes is the lowest score, with a .8 overlap
        # with item 1 so it get suppressed
        a = [0., 0., 5., 10.]
        b = [0., 0., 4., 10.]
        c = [5., 5., 10., 10.]
        assert Bboxer.single_bb_iou(a, b) == .8
        assert Bboxer.single_bb_iou(a, c) == 0.
        boxes = torch.tensor([a,b,c])
        scores = torch.tensor([.1, .2, .3])
        assert boxes.shape[0] == scores.shape[0]

        ret_keep, ret_count = Predict.nms(boxes, scores)

        self.assert_arr_equals(ret_keep, [2, 1, 0])
        assert ret_count == 2

    def test_nms__doesnt_suppress_if_under_overlap_thresh(self):
        # item 0 in boxes is the lowest score, with a .8 overlap
        # but overlap_thres is .1 so it isn't suppresed
        a = [0., 0., 5., 10.]
        b = [0., 0., 4., 10.]
        c = [5., 5., 10., 10.]
        overlap = 0.8
        assert Bboxer.single_bb_iou(a, b) == overlap
        assert Bboxer.single_bb_iou(a, c) == 0.
        boxes = torch.tensor([a,b,c])
        scores = torch.tensor([.1, .2, .3])
        assert boxes.shape[0] == scores.shape[0]

        # overlap suppresses
        ret_keep, ret_count = Predict.nms(boxes, scores, overlap=0.79)
        self.assert_arr_equals(ret_keep, [2, 1, 0])
        assert ret_count == 2

        # overlap allows
        ret_keep, ret_count = Predict.nms(boxes, scores, overlap=overlap)
        self.assert_arr_equals(ret_keep, [2, 1, 0])
        assert ret_count == 3
