import unittest

from ssdmultibox.predict import Predict
from tests.mixins import ModelAndDatasetSetupMixin


class PredictTests(ModelAndDatasetSetupMixin, unittest.TestCase):

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
