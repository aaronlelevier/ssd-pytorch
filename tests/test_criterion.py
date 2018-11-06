import unittest

import torch
from torch.utils.data import DataLoader

from ssdmultibox import criterion
from ssdmultibox.datasets import TrainPascalDataset
from ssdmultibox.models import SSDModel, vgg16_bn

PREDS = None

def get_preds(ims):
    global PREDS

    if PREDS:
        return PREDS

    vgg_base = vgg16_bn(pretrained=True)

    # freeze base network
    for layer in vgg_base.parameters():
        layer.requires_grad = False

    model = SSDModel(vgg_base)

    PREDS = model(ims)
    return PREDS


class CriterionTests(unittest.TestCase):

    def test_bbs_loss(self):
        dataset = TrainPascalDataset()
        dataloader = DataLoader(dataset, batch_size=4, num_workers=0)
        image_ids, ims, gt_bbs, gt_cats = next(iter(dataloader))
        preds = get_preds(ims)
        bbs_criterion = criterion.BbsL1Loss()

        ret = bbs_criterion(gt_bbs, gt_cats, preds)

        assert isinstance(ret, torch.Tensor)
        assert ret.item() > 0
