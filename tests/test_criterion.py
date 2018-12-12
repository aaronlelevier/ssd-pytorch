import torch

from unittest.mock import patch
from ssdmultibox import criterion
from ssdmultibox.config import cfg
from tests.base import ModelAndDatasetBaseTestCase


class CriterionTests(ModelAndDatasetBaseTestCase):

    @staticmethod
    def assert_is_loss_tensor(loss):
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_ssd_loss(self):
        ssd_criterion = criterion.SSDLoss()
        targets = (self.gt_bbs, self.gt_cats)

        ssd_loss, loc_loss, conf_loss = ssd_criterion(self.preds, targets)

        self.assert_is_loss_tensor(ssd_loss)
        self.assert_is_loss_tensor(loc_loss)
        self.assert_is_loss_tensor(conf_loss)

    def test_bbs_loss(self):
        bbs_criterion = criterion.BbsL1Loss()
        bbs_preds, cats_preds = self.preds
        targets = (self.gt_bbs, self.gt_cats)

        ret = bbs_criterion(bbs_preds, targets)

        self.assert_is_loss_tensor(ret)

    def test_cats_loss(self):
        cats_criterion = criterion.CatsBCELoss()
        bbs_preds, cats_preds = self.preds

        ret = cats_criterion(cats_preds, self.gt_cats)

        self.assert_is_loss_tensor(ret)

    def test_one_hot_encoding(self):
        y = torch.LongTensor(4,9).random_() % cfg.NUM_CLASSES
        assert y.shape == (4, 9)

        ret = criterion.CatsBCELoss.one_hot_encoding(y)

        assert ret.shape == (4, 21)

    def test_one_hot_encoding_unsqueezes_if_input_is_1d(self):
        y = torch.LongTensor(4).random_() % cfg.NUM_CLASSES
        assert y.shape == (4,)

        ret = criterion.CatsBCELoss.one_hot_encoding(y)

        assert ret.shape == (4, 21)


class CatsBCELossTests(ModelAndDatasetBaseTestCase):

    def setUp(self):
        super().setUp()

        self.cats_criterion = criterion.CatsBCELoss()
        self.bbs_preds, self.cats_preds = self.preds

    @patch("ssdmultibox.criterion.F.binary_cross_entropy_with_logits")
    @patch("ssdmultibox.criterion.torch.cat")
    def test_get_neg_loss_called_wo_bg(self, mock_torch_cat, mock_bce_loss):
        pos_idxs = self.gt_cats != 20
        total_count = self.cats_preds.shape[0] * self.cats_preds.shape[1]
        pos_count = pos_idxs.sum().item()
        neg_count = total_count - pos_count
        mock_bce_loss.side_effect = [
            torch.randn(neg_count, 20), torch.randn(pos_count, 20)]

        self.cats_criterion(self.cats_preds, self.gt_cats)

        assert mock_bce_loss.call_count == 2
        # neg_loss
        assert mock_bce_loss.call_args_list[0][1]['input'].shape == \
            (neg_count, 20)
        assert mock_bce_loss.call_args_list[0][1]['target'].shape == \
            (neg_count, 20)
        assert mock_bce_loss.call_args_list[0][1]['target'].sum().item() == 0
        # pos_loss
        assert mock_bce_loss.call_args_list[1][1]['input'].shape == \
            (pos_count, 20)
        assert mock_bce_loss.call_args_list[1][1]['target'].shape == \
            (pos_count, 20)
        # torch.cat
        assert mock_torch_cat.called
        assert mock_torch_cat.call_args[0][0][0].shape == (pos_count, 20)
        assert self.cats_criterion.hard_mining_ratio == cfg.HARD_MINING_RATIO
        assert mock_torch_cat.call_args[0][0][1].shape == \
            (pos_count * self.cats_criterion.hard_mining_ratio, 20)
