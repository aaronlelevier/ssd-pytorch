import unittest
from unittest.mock import patch

import pytest

from ssdmultibox.predict import Predict
from tests.mixins import ModelAndDatasetSetupMixin


class PredictTests(ModelAndDatasetSetupMixin, unittest.TestCase):

    @patch("ssdmultibox.predict.Predict.single_predict")
    def test_detections_for_single_category__calls_single_predict(
            self, mock_single_predict):
        cls_id = 6

        ret = Predict.detections_for_single_category(cls_id, self.preds)

        assert mock_single_predict.called
        assert mock_single_predict.call_args[0][0] == cls_id

    def test_detections_for_single_category__returns_correct_shapes(self):
        cls_id = 12

        ret = Predict.detections_for_single_category(cls_id, self.preds)

        # same shapes from single_predict, need `if` because sometimes
        # this is none, but if it's detected something, I want check shapes
        if ret:
            ret_bbs , ret_scores = ret
            assert len(ret_bbs.shape) == 2
            assert len(ret_scores.shape) == 1
            assert ret_bbs.shape[0] == ret_scores.shape[0]

    def test_detections_for_single_category__explicit_choose_item_in_batch(self):
        cls_id = 12

        ret = Predict.detections_for_single_category(cls_id, self.preds)
        ret2 = Predict.detections_for_single_category(cls_id, self.preds, index=1)

        if ret and ret2:
            assert ret[0].shape != ret2[0].shape, \
            "shouldn't be the same because different training example items"

    def test_detections_for_single_category__bad_index_raises_error(self):
        cls_id = 12
        index = 5
        assert self.preds[0][0][0].shape[0] < index

        with pytest.raises(IndexError):
            Predict.detections_for_single_category(cls_id, self.preds, index=index)

    def test_get_stacked_bbs_and_cats(self):
        ret = Predict.get_stacked_bbs_and_cats(self.preds)

        assert len(ret) == 2

        assert ret[0].shape == (4, 11640, 4)
        assert ret[1].shape == (4, 11640, 20)

    def test_single_predict(self):
        cls_id = 0
        bbs, cats = Predict.get_stacked_bbs_and_cats(self.preds)
        item_cats = cats[0]
        item_bbs = bbs[0]

        ret_bbs, ret_scores = Predict.single_predict(cls_id, item_bbs, item_cats)

        assert len(ret_bbs.shape) == 2
        assert len(ret_scores.shape) == 1
        assert ret_bbs.shape[0] == ret_scores.shape[0]
        assert ret_bbs.shape[1] == 4, "4 points p/ bb"
        assert ret_scores.sum().item() != 0, "pred confidence should always be greater than 0"
