import torch
import torch.nn.functional as F
from torch import nn

from ssdmultibox.bboxer import TensorBboxer
from ssdmultibox.config import cfg
from ssdmultibox.datasets import NUM_CLASSES, SIZE, device


class CatsBCELoss(nn.Module):
    def __init__(self, hard_mining_ratio=3):
        super().__init__()
        self.hard_mining_ratio = hard_mining_ratio

    def forward(self, inputs, targets):
        """
        Calculates the categorical loss of the gt_cats and hard negative mining
        """
        pos_idxs = targets != 20
        neg_idxs = targets == 20
        neg_loss = F.cross_entropy(inputs[neg_idxs], targets[neg_idxs], reduction='none')
        neg_loss_sorted, _ = neg_loss.sort(descending=True)
        neg_hard_mining_count = pos_idxs.sum().item() * self.hard_mining_ratio
        neg_hard_mining_loss = neg_loss_sorted[:neg_hard_mining_count]
        pos_loss = F.cross_entropy(inputs[pos_idxs], targets[pos_idxs], reduction='none')
        print('pos_loss: {:.4f} neg_hard_mining_loss: {:.4f}'.format(
            pos_loss.sum().item(), neg_hard_mining_loss.sum().item()))
        return torch.cat((pos_loss, neg_hard_mining_loss)).sum()

    @staticmethod
    def one_hot_encoding(y):
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_onehot = torch.FloatTensor(y.shape[0], NUM_CLASSES).to(device)
        y_onehot.zero_()
        y = y.type(torch.long)
        return y_onehot.scatter_(1, y, 1)


class BbsL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stacked_anchor_boxes = TensorBboxer.get_stacked_anchor_boxes()

    def forward(self, inputs, targets):
        preds = inputs
        gt_bbs, gt_cats = targets
        preds_w_offsets =  self.stacked_anchor_boxes + preds
        gt_idxs = gt_cats != 20
        inputs = torch.clamp(preds_w_offsets[gt_idxs], min=0, max=SIZE)
        targets = gt_bbs[gt_idxs].type(inputs.dtype)
        return F.smooth_l1_loss(inputs, targets, reduction='sum')


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
        n = (gt_cats != 20).sum().type(conf.dtype).to(device)
        print('n: {} bbs_loss: {:.4f} cats_loss: {:.4f}'.format(
            n.item(), loc.item(), conf.item()))
        # TODO: added addit returns of loc, conf losses for debugging
        return (1/n) * (conf+loc), loc, conf
