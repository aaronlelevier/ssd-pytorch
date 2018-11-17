import unittest

import numpy as np
from torch.utils.data import DataLoader

from ssdmultibox.datasets import BATCH, TrainPascalDataset
from ssdmultibox.models import SSDModel


class BaseTestCase(unittest.TestCase):

    def assert_arr_equals(self, ret, raw_ret):
        assert np.isclose(
                np.array(ret, dtype=np.float16),
                np.array(raw_ret, dtype=np.float16)
            ).all(), f"\nret:\n{ret}\nraw_ret:\n{raw_ret}"


PREDS_CACHE = {}

class ModelAndDatasetBaseTestCase(BaseTestCase):
    """
    Test class for any tests that need model predictions

    Predictions cached in PREDS_CACHE so they can be reused
    without having to re-invoke the model
    """

    @classmethod
    def setUpClass(cls):
        if not PREDS_CACHE:
            # data
            dataset = TrainPascalDataset()
            dataloader = DataLoader(dataset, batch_size=BATCH, num_workers=0)
            image_ids, ims, gt_bbs, gt_cats = next(iter(dataloader))
            ims, gt_bbs, gt_cats = dataset.to_device(ims, gt_bbs, gt_cats)
            # model
            model = SSDModel()
            preds = model(ims)
            PREDS_CACHE.update({
                'gt_bbs': gt_bbs,
                'gt_cats': gt_cats,
                'preds': preds
            })
        for k,v in PREDS_CACHE.items():
            setattr(cls, k, v)

    def setUp(self):
        self.gt_bbs = self.__class__.gt_bbs
        self.gt_cats = self.__class__.gt_cats
        self.preds = self.__class__.preds
