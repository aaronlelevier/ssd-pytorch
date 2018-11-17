import unittest

import torch

from ssdmultibox import criterion
from ssdmultibox.datasets import NUM_CLASSES
from tests.base import ModelAndDatasetBaseTestCase


class CriterionTests(ModelAndDatasetBaseTestCase):

    def test_bbs_loss(self):
        bbs_criterion = criterion.BbsL1Loss()
        inputs = self.preds
        targets = (self.gt_bbs, self.gt_cats)

        ret = bbs_criterion(inputs, targets)

        assert isinstance(ret, torch.Tensor)
        assert ret.item() > 0

    def test_cats_loss(self):
        cats_criterion = criterion.CatsBCELoss()
        inputs = self.preds
        targets = self.gt_cats

        ret = cats_criterion(inputs, targets)

        assert isinstance(ret, torch.Tensor)
        assert ret.item() > 0

    def test_ssd_loss(self):
        ssd_criterion = criterion.SSDLoss()
        inputs = self.preds
        targets = (self.gt_bbs, self.gt_cats)

        ret = ssd_criterion(inputs, targets)

        assert isinstance(ret, torch.Tensor)
        assert ret.item() > 0

    def test_one_hot_encoding(self):
        y = torch.LongTensor(4,9).random_() % NUM_CLASSES
        assert y.shape == (4, 9)

        ret = criterion.CatsBCELoss.one_hot_encoding(y)

        assert ret.shape == (4, 9, 21)

    def test_one_hot_encoding_unsqueezes_if_input_is_1d(self):
        y = torch.LongTensor(4).random_() % NUM_CLASSES
        assert y.shape == (4,)

        ret = criterion.CatsBCELoss.one_hot_encoding(y)

        assert ret.shape == (4, 1, 21)
