import unittest

from tests.base import ModelAndDatasetBaseTestCase


class SSDModelTests(ModelAndDatasetBaseTestCase):

    def test_model_feature_map_bbs_output_sizes(self):
        ret = [tuple(self.preds[i][0][0].shape) for i in range(6)]

        assert ret == [
            (4, 5776),
            (4, 1444),
            (4, 400),
            (4, 100),
            (4, 36),
            (4, 4)
        ]

    def test_model_feature_map_cats_output_sizes(self):
        ret = [tuple(self.preds[i][0][1].shape) for i in range(6)]

        assert ret == [
            (4, 30324),
            (4, 7581),
            (4, 2100),
            (4, 525),
            (4, 189),
            (4, 21)
        ]

    def test_model_1st_layer_feature_map_bbs_output_sizes(self):
        ret = [tuple(self.preds[0][i][0].shape) for i in range(6)]

        assert ret == [(4, 5776) for i in range(6)]

    def test_model_1st_layer_feature_map_cats_output_sizes(self):
        ret = [tuple(self.preds[0][i][1].shape) for i in range(6)]

        assert ret == [(4, 30324) for i in range(6)]
