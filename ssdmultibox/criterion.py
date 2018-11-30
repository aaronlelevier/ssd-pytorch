import torch
import torch.nn.functional as F
from torch import nn

from ssdmultibox.datasets import NUM_CLASSES, SIZE, device, Bboxer


class CatsBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        batch_size = targets.shape[0]
        cats_preds = inputs.reshape(batch_size, -1, NUM_CLASSES)[:,:,:-1]
        gt_idxs = targets != 20
        inputs = cats_preds[gt_idxs]
        targets = targets[gt_idxs]
        one_hot_targets = self.one_hot_encoding(targets)[:,:-1]
        return F.binary_cross_entropy_with_logits(inputs, one_hot_targets, reduction='sum')

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

    def forward(self, inputs, targets):
        preds = inputs
        gt_bbs, gt_cats = targets
        stacked_anchor_boxes = torch.tensor(
            Bboxer.get_stacked_anchor_boxes(), dtype=preds.dtype).to(device)
        preds_w_offsets =  stacked_anchor_boxes + preds
        gt_idxs = gt_cats != 20
        # clamp - b/c these are the bounds of our bbs prediction
        targets = torch.clamp(gt_bbs[gt_idxs].type(inputs.dtype), min=0, max=1)
        inputs = preds_w_offsets[gt_idxs]
        return F.smooth_l1_loss(inputs, targets, reduction='sum')


class SSDLoss(nn.Module):
    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha
        self.bbs_loss = BbsL1Loss()
        self.cats_loss = CatsBCELoss()

    def forward(self, inputs, targets):
        bbs_preds, cats_preds = inputs
        gt_bbs, gt_cats = targets
        conf = self.cats_loss(cats_preds, gt_cats)
        loc = self.bbs_loss(bbs_preds, (gt_bbs, gt_cats))
        n = (gt_cats != 20).sum().type(conf.dtype).to(device)
        print('bbs_loss:', loc.item())
        print('cats_loss:', conf.item())
        # TODO: added addit returns of loc, conf losses for debugging
        return (1/n) * (conf + (self.alpha*loc)), loc, conf
