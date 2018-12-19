from ssdmultibox import plotting
from tests.base import ModelAndDatasetBaseTestCase


class PlottingTests(ModelAndDatasetBaseTestCase):

    def test_get_anchor_bbs_targets(self):
        idx = 0

        bbs, cats = plotting.get_anchor_bbs_targets(idx, self.gt_cats)

        self.assert_arr_equals(
            bbs,
            [[0.33333334, 0.33333334, 0.6666667 , 0.6666667 ],
            [0.16666667, 0.33333334, 0.8333333 , 0.6666667 ],
            [0.33333334, 0.33333334, 1.        , 0.6666667 ],
            [0.3243179 , 0.33333334, 0.67568207, 0.6666667 ]]
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
