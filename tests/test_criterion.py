import unittest

import torch

from ssdmultibox import criterion
from ssdmultibox.config import cfg
from ssdmultibox.datasets import NUM_CLASSES
from tests.base import ModelAndDatasetBaseTestCase


class CriterionTests(ModelAndDatasetBaseTestCase):

    def test_bbs_loss(self):
        bbs_criterion = criterion.BbsL1Loss()
        bbs_preds, cats_preds = self.preds
        targets = (self.gt_bbs, self.gt_cats)

        ret = bbs_criterion(bbs_preds, targets)

        assert isinstance(ret, torch.Tensor)
        assert ret.item() > 0

    def test_bbs_loss__anchor_boxes_are_normalized_by_size(self):
        bbs_criterion = criterion.BbsL1Loss()

        # the last anchor box is a feature map cell of size 1
        # so it's dimentions should be the NORMALIZED_SIZE
        self.assert_arr_equals(
            bbs_criterion.stacked_anchor_boxes[-1],
            [0, 0, cfg.NORMALIZED_SIZE, cfg.NORMALIZED_SIZE]
        )

    def test_cats_loss(self):
        cats_criterion = criterion.CatsBCELoss()
        bbs_preds, cats_preds = self.preds

        ret = cats_criterion(cats_preds, self.gt_cats)

        assert isinstance(ret, torch.Tensor)
        assert ret.item() > 0

    def test_ssd_loss(self):
        ssd_criterion = criterion.SSDLoss()
        targets = (self.gt_bbs, self.gt_cats)

        ssd_loss, loc_loss, conf_loss = ssd_criterion(self.preds, targets)

        assert isinstance(ssd_loss, torch.Tensor)
        assert ssd_loss.item() > 0

    def test_one_hot_encoding(self):
        y = torch.LongTensor(4,9).random_() % NUM_CLASSES
        assert y.shape == (4, 9)

        ret = criterion.CatsBCELoss.one_hot_encoding(y)

        assert ret.shape == (4, 21)

    def test_one_hot_encoding_unsqueezes_if_input_is_1d(self):
        y = torch.LongTensor(4).random_() % NUM_CLASSES
        assert y.shape == (4,)

        ret = criterion.CatsBCELoss.one_hot_encoding(y)

        assert ret.shape == (4, 21)
