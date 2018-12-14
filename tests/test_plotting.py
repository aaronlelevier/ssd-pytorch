from tests.base import ModelAndDatasetBaseTestCase
from ssdmultibox import plotting


class PlottingTests(ModelAndDatasetBaseTestCase):

    def test_get_anchor_bbs_targets(self):
        idx = 0

        bbs, cats = plotting.get_anchor_bbs_targets(idx, self.gt_cats)

        self.assert_arr_equals(
            bbs,
            [[100.0000, 100.0000, 200.0000, 200.0000],
             [ 50.0000, 100.0000, 250.0000, 200.0000],
             [100.0000, 100.0000, 300.0000, 200.0000],
             [ 97.2954, 100.0000, 202.7046, 200.0000]]
        )
        self.assert_arr_equals(
            cats,
            [6, 6, 6, 6]
        )

    def test_get_pred_targets(self):
        idx = 0

        bbs, cats = plotting.get_pred_targets(idx, self.gt_cats, self.preds)

        assert bbs.shape == (4, 4)
        assert not (cats.numpy() == [6,6,6,6]).all()
