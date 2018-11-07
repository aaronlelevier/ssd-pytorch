import unittest

import torch
from torch.utils.data import DataLoader

from ssdmultibox import criterion
from ssdmultibox.datasets import TrainPascalDataset
from ssdmultibox.models import SSDModel, vgg16_bn


class CriterionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # data
        dataset = TrainPascalDataset()
        dataloader = DataLoader(dataset, batch_size=10, num_workers=0)
        image_ids, ims, cls.gt_bbs, cls.gt_cats = next(iter(dataloader))
        # model
        vgg_base = vgg16_bn(pretrained=True)
        for layer in vgg_base.parameters():
            layer.requires_grad = False
        model = SSDModel(vgg_base)
        cls.preds = model(ims)

    def setUp(self):
        self.gt_bbs = self.__class__.gt_bbs
        self.gt_cats = self.__class__.gt_cats
        self.preds = self.__class__.preds

    def test_bbs_loss(self):
        bbs_criterion = criterion.BbsL1Loss()

        ret = bbs_criterion(self.gt_bbs, self.gt_cats, self.preds)

        assert isinstance(ret, torch.Tensor)
        assert ret.item() > 0

    def test_cats_loss(self):
        cats_criterion = criterion.CatsBCELoss()

        ret = cats_criterion(self.gt_cats, self.preds)

        assert isinstance(ret, torch.Tensor)
        assert ret.item() > 0

    def test_ssd_loss(self):
        ssd_criterion = criterion.SSDLoss()

        ret = ssd_criterion(self.gt_bbs, self.gt_cats, self.preds)

        assert isinstance(ret, torch.Tensor)
        assert ret.item() > 0
