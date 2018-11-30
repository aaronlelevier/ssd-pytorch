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
        return F.cross_entropy(
            cats_preds[gt_idxs], targets[gt_idxs], reduction='sum')

    # NOTE: not in use, but if we change to F.binary_cross_entropy_with_logits
    # then it's needed
    @staticmethod
    def one_hot_encoding(y):
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_onehot = torch.FloatTensor(y.shape[0], y.shape[1], NUM_CLASSES).to(device)
        y_onehot.zero_()
        y = y.type(torch.long)
        # expand from shape (4, 1444) to (4, 1444, 1) for ex to work w/ `scatter_`
        y = y.unsqueeze(-1)
        return y_onehot.scatter_(2, y, 1)


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
        targets = gt_bbs[gt_idxs]
        inputs = preds_w_offsets[gt_idxs]
        return F.smooth_l1_loss(inputs, targets.type(inputs.dtype), reduction='sum')


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
        return (1/n) * (conf + (self.alpha*loc))
