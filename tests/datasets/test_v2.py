import numpy as np

from ssdmultibox.datasets import SIZE
from ssdmultibox.datasets.v2 import TrainPascalDataset
from tests.base import BaseTestCase

np.set_printoptions(precision=15)

TEST_IMAGE_ID = 17


class PascalDatasetTestsV2(BaseTestCase):

    def setUp(self):
        self.dataset = TrainPascalDataset()
        # so we don't have to build the entire annotations cache
        self.dataset.get_annotations()

    def test_getitem(self):
        ret_image_id, ret_im, ret_gt_bbs, ret_gt_cats = self.dataset[1]

        assert ret_image_id == TEST_IMAGE_ID
        assert ret_im.shape == (3, SIZE, SIZE)
        self.assert_arr_equals(
            ret_gt_bbs,
            np.array([[ 50.27472527472527, 115.00000000000001, 163.010989010989  ,
                    173.37500000000003],
                [ 63.46153846153846,  55.625           , 275.92307692307696,
                    250.87500000000003]])
        )
        self.assert_arr_equals(ret_gt_cats, [14, 12])
