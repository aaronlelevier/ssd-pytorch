import torch
import torch.nn.functional as F
from torch import nn

from ssdmultibox.bboxer import TensorBboxer
from ssdmultibox.config import cfg


class SSDLoss(nn.Module):
    def __init__(self, alpha=cfg.SSD_LOSS_ALPHA):
        super().__init__()
        self.alpha = alpha
        self.bbs_loss = BbsL1Loss()
        self.cats_loss = CatsBCELoss()

    def forward(self, inputs, targets):
        bbs_preds, cats_preds = inputs
        gt_bbs, gt_cats = targets
        conf = self.cats_loss(cats_preds, gt_cats)
        loc = self.bbs_loss(bbs_preds, (gt_bbs, gt_cats)) * self.alpha
        n = (gt_cats != 20).sum().type(conf.dtype).to(cfg.DEVICE)
        print('n: {} bbs_loss: {:.4f} cats_loss: {:.4f}'.format(
            n.item(), loc.item(), conf.item()))
        # TODO: added addit returns of loc, conf losses for debugging
        return (1/n) * (conf+loc), loc, conf


class BbsL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        bbs_preds = inputs
        gt_bbs, gt_cats = targets
        gt_idxs = gt_cats != 20
        pos_inputs = bbs_preds[gt_idxs]
        pos_targets = gt_bbs[gt_idxs].type(inputs.dtype)
        return F.smooth_l1_loss(pos_inputs, pos_targets, reduction='sum')


class CatsBCELoss(nn.Module):
    def __init__(self, hard_mining_ratio=cfg.HARD_MINING_RATIO):
        super().__init__()
        self.hard_mining_ratio = hard_mining_ratio

    def forward(self, inputs, targets):
        """
        Calculates the categorical loss of the gt_cats and hard negative mining
        """
        pos_idxs = targets != 20
        neg_idxs = targets == 20
        neg_loss = F.binary_cross_entropy_with_logits(
            input=inputs[neg_idxs][:,:-1],
            target=self.one_hot_encode_to_zeros(targets[neg_idxs])[:,:-1],
            reduction='none'
        )

        neg_loss_sorted, _ = neg_loss.sort(descending=True)
        neg_hard_mining_count = pos_idxs.sum().item() * self.hard_mining_ratio
        neg_hard_mining_loss = neg_loss_sorted[:neg_hard_mining_count]
        pos_loss = F.binary_cross_entropy_with_logits(
            input=inputs[pos_idxs][:,:-1],
            target=self.one_hot_encoding(targets[pos_idxs])[:,:-1],
            reduction='none'
        )

        print('pos_loss: {:.4f} neg_hard_mining_loss: {:.4f}'.format(
            pos_loss.sum().item(), neg_hard_mining_loss.sum().item()))

        return torch.cat((pos_loss, neg_hard_mining_loss)).sum()

    @staticmethod
    def one_hot_encoding(y):
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_onehot = torch.FloatTensor(y.shape[0], cfg.NUM_CLASSES).to(cfg.DEVICE)
        y_onehot.zero_()
        y = y.type(torch.long)
        return y_onehot.scatter_(1, y, 1)

    @staticmethod
    def one_hot_encode_to_zeros(y):
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_onehot = torch.FloatTensor(y.shape[0], cfg.NUM_CLASSES).to(cfg.DEVICE)
        return y_onehot.zero_()
