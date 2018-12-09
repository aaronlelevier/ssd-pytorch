import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from ssdmultibox.datasets import BATCH, TrainPascalFlatDataset
from ssdmultibox.models import SSDModel


class BaseTestCase(unittest.TestCase):

    def assert_arr_equals(self, ret, raw_ret, msg=""):
        """
        User to assert arrays are equal when precision is an issue
        """
        error_msg = f"\nret:\n{ret}\nraw_ret:\n{raw_ret}"
        if msg:
            error_msg += f"\n{msg}"

        assert np.isclose(
                np.array(ret, dtype=np.float16),
                np.array(raw_ret, dtype=np.float16)
            ).all(), error_msg

    def assert_float_equals(self, ret, raw_ret, msg=""):
        """
        User to assert floats are equal when precision is an issue
        """
        def str_float(x):
            return "{:.8f}".format(x)

        if isinstance(ret, torch.Tensor):
            ret = ret.item()

        assert str_float(ret) == str_float(raw_ret), msg


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
            dataset = TrainPascalFlatDataset()
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
