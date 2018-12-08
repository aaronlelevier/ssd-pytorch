from unittest.mock import patch

import pytest
import torch

from ssdmultibox.bboxer import Bboxer
from ssdmultibox.config import cfg
from ssdmultibox.predict import Predict
from tests.base import BaseTestCase, ModelAndDatasetBaseTestCase


class PredictTests(ModelAndDatasetBaseTestCase):

    def test_predict_all(self):
        single_preds = None
        for c in range(cfg.NUM_CLASSES):
            single_preds = Predict.single_predict(c, self.preds)
            if single_preds:
                break

        ret_bbs, ret_scores, ret_cls_ids = Predict.predict_all(self.preds)

        single_bbs, single_scores, single_cls_ids = single_preds
        # more then 1 cls of obj detected
        assert ret_bbs.shape[0] > single_bbs.shape[0]
        assert len(ret_bbs.shape) == 2
        assert ret_bbs.shape[1] == 4
        assert ret_bbs.shape[0] == ret_scores.shape[0]
        assert ret_bbs.shape[0] == ret_cls_ids.shape[0]

    @patch("ssdmultibox.predict.Predict.single_nms")
    def test_single_predict__calls_single_nms(
            self, mock_single_nms):
        cls_id = 6
        conf_thresh = 0.5

        Predict.single_predict(cls_id, self.preds, conf_thresh=conf_thresh)

        assert mock_single_nms.called
        assert mock_single_nms.call_args[0][0] == cls_id
        assert len(mock_single_nms.call_args[0][1][0]) == 4
        assert mock_single_nms.call_args[0][-1] == conf_thresh

    def test_single_predict__returns_correct_shapes(self):
        cls_id = 12

        ret = Predict.single_predict(cls_id, self.preds)

        # same shapes from single_nms, need `if` because sometimes
        # this is none, but if it's detected something, I want check shapes
        if ret:
            ret_bbs, ret_scores, ret_cls_ids = ret
            assert len(ret_bbs.shape) == 2
            assert len(ret_scores.shape) == 1
            assert ret_bbs.shape[0] == ret_scores.shape[0]
            assert ret_cls_ids.shape[0] == ret_scores.shape[0]

    def test_single_predict__explicit_choose_item_in_batch(self):
        cls_id = 12

        ret = Predict.single_predict(cls_id, self.preds)
        ret2 = Predict.single_predict(cls_id, self.preds, index=1)

        if ret and ret2:
            assert ret[0].shape != ret2[0].shape, \
            "shouldn't be the same because different training example items"

    def test_single_predict__bad_index_raises_error(self):
        cls_id = 12
        index = 5
        assert self.preds[0][0][0].shape[0] < index

        with pytest.raises(IndexError):
            Predict.single_predict(cls_id, self.preds, index=index)

    def test_single_nms(self):
        cls_id = 0
        bbs, cats = self.preds
        item_cats = cats[0]
        item_bbs = bbs[0]

        ret_bbs, ret_scores, ret_cls_ids = Predict.single_nms(cls_id, item_bbs, item_cats)

        assert len(ret_bbs.shape) == 2
        assert len(ret_scores.shape) == 1
        assert ret_bbs.shape[0] == ret_scores.shape[0]
        assert ret_bbs.shape[1] == 4, "4 points p/ bb"
        assert ret_scores.sum().item() != 0, "pred confidence should always be greater than 0"
        assert ret_cls_ids.shape[0] == ret_scores.shape[0]


@pytest.mark.unit
class PredictUnitTests(BaseTestCase):

    def test_single_nms__both_returned_bc_pass_conf_thresh_and_no_overlap(self):
        cls_id = 0
        a = [0., 0., 5., 10.]
        b = [5., 5., 10., 10.]
        assert Bboxer.single_bb_iou(a, b) == 0
        item_bbs = torch.tensor([a, b])
        assert list(item_bbs.shape) == [2, 4]
        cats_a = torch.cat((torch.tensor([.5]), torch.zeros(19)))
        cats_b = torch.cat((torch.tensor([.6]), torch.zeros(19)))
        item_cats = torch.stack((cats_a, cats_b))
        assert list(item_cats.shape) == [2, 20]

        ret_bbs, ret_scores, ret_cls_ids = Predict.single_nms(cls_id, item_bbs, item_cats)

        self.assert_arr_equals(ret_bbs, [b, a])
        self.assert_arr_equals(ret_scores, [.6, .5])
        self.assert_arr_equals(ret_cls_ids, [0, 0])

    def test_single_nms__one_returned_bc_pass_conf_thresh_too_low(self):
        cls_id = 0
        a = [0., 0., 5., 10.]
        b = [5., 5., 10., 10.]
        assert Bboxer.single_bb_iou(a, b) == 0
        item_bbs = torch.tensor([a, b])
        assert list(item_bbs.shape) == [2, 4]
        low_conf_thresh = .09
        assert low_conf_thresh < cfg.NMS_CONF_THRESH
        cats_a = torch.cat((torch.tensor([low_conf_thresh]), torch.zeros(19)))
        assert (cats_a > cfg.NMS_CONF_THRESH).sum().item() == 0
        cats_b = torch.cat((torch.tensor([.6]), torch.zeros(19)))
        item_cats = torch.stack((cats_a, cats_b))
        assert list(item_cats.shape) == [2, 20]

        ret_bbs, ret_scores, ret_cls_ids = Predict.single_nms(cls_id, item_bbs, item_cats)

        self.assert_arr_equals(ret_bbs, [b])
        self.assert_arr_equals(ret_scores, [.6])
        self.assert_arr_equals(ret_cls_ids, [0])

    def test_single_nms__can_change_conf_thresh(self):
        conf_thresh = 0.55
        cls_id = 0
        a = [0., 0., 5., 10.]
        b = [5., 5., 10., 10.]
        assert Bboxer.single_bb_iou(a, b) == 0
        item_bbs = torch.tensor([a, b])
        assert list(item_bbs.shape) == [2, 4]
        cats_a = torch.cat((torch.tensor([.5]), torch.zeros(19)))
        assert (cats_a > conf_thresh).sum().item() == 0
        cats_b = torch.cat((torch.tensor([.6]), torch.zeros(19)))
        assert (cats_b > conf_thresh).sum().item() == 1
        item_cats = torch.stack((cats_a, cats_b))
        assert list(item_cats.shape) == [2, 20]

        ret_bbs, ret_scores, ret_cls_ids = Predict.single_nms(
            cls_id, item_bbs, item_cats, conf_thresh)

        self.assert_arr_equals(ret_bbs, [b])
        self.assert_arr_equals(ret_scores, [.6])
        self.assert_arr_equals(ret_cls_ids, [0])

    def test_single_nms__filters_out_by_cls_id(self):
        cls_id = 0
        a = [0., 0., 5., 10.]
        b = [5., 5., 10., 10.]
        assert Bboxer.single_bb_iou(a, b) == 0
        item_bbs = torch.tensor([a, b])
        assert list(item_bbs.shape) == [2, 4]
        # the ordering of the one-hot category max'es dictates which
        # cls_id, so cats_a is cls 1, cats_b is cls 0
        cats_a = torch.cat((torch.tensor([0, .5]), torch.zeros(18)))
        cats_b = torch.cat((torch.tensor([.6, 0]), torch.zeros(18)))
        item_cats = torch.stack((cats_a, cats_b))
        assert list(item_cats.shape) == [2, 20]

        ret_bbs, ret_scores, ret_cls_ids = Predict.single_nms(cls_id, item_bbs, item_cats)

        self.assert_arr_equals(ret_bbs, [b])
        self.assert_arr_equals(ret_scores, [.6])
        self.assert_arr_equals(ret_cls_ids, [0])

    def test_nms__suppresses_overlapping_bb_with_lower_score(self):
        # item 0 in boxes is the lowest score, with a .8 overlap
        # with item 1 so it get suppressed
        a = [0., 0., 5., 10.]
        b = [0., 0., 4., 10.]
        c = [5., 5., 10., 10.]
        assert Bboxer.single_bb_iou(a, b) == .8
        assert Bboxer.single_bb_iou(a, c) == 0.
        boxes = torch.tensor([a,b,c])
        scores = torch.tensor([.1, .2, .3])
        assert boxes.shape[0] == scores.shape[0]

        ret_keep, ret_count = Predict.nms(boxes, scores)

        self.assert_arr_equals(ret_keep, [2, 1, 0])
        assert ret_count == 2

    def test_nms__doesnt_suppress_if_under_overlap_thresh(self):
        # item 0 in boxes is the lowest score, with a .8 overlap
        # but overlap_thres is .1 so it isn't suppresed
        a = [0., 0., 5., 10.]
        b = [0., 0., 4., 10.]
        c = [5., 5., 10., 10.]
        overlap = 0.8
        assert Bboxer.single_bb_iou(a, b) == overlap
        assert Bboxer.single_bb_iou(a, c) == 0.
        boxes = torch.tensor([a,b,c])
        scores = torch.tensor([.1, .2, .3])
        assert boxes.shape[0] == scores.shape[0]

        # overlap suppresses
        ret_keep, ret_count = Predict.nms(boxes, scores, overlap=0.79)
        self.assert_arr_equals(ret_keep, [2, 1, 0])
        assert ret_count == 2

        # overlap allows
        ret_keep, ret_count = Predict.nms(boxes, scores, overlap=overlap)
        self.assert_arr_equals(ret_keep, [2, 1, 0])
        assert ret_count == 3
