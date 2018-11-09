import torch
import torch.nn.functional as F
from torch import nn

from ssdmultibox.datasets import NUM_CLASSES, SIZE, Bboxer, device


class CatsBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        preds = inputs
        gt_cats = targets
        loss = torch.tensor(0, dtype=torch.float32).to(device)

        for fm_idx in range(len(preds)):
            for ar_idx in range(len(preds[fm_idx])):
                loss.add_(
                    self._cats_loss(gt_cats[fm_idx][ar_idx], preds[fm_idx][ar_idx][1]))
        return loss

    def _cats_loss(self, y, yhat):
        batch_size = y.shape[0]
        cats_label = Bboxer.one_hot_encoding(y)[:,:,:-1]
        cats_preds = yhat.reshape(batch_size, -1, NUM_CLASSES)[:,:,:-1]
        gt_idxs = y != 20
        return F.binary_cross_entropy_with_logits(
            cats_label[gt_idxs], cats_preds[gt_idxs], reduction='sum')


class BbsL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        preds = inputs
        gt_bbs, gt_cats = targets
        loss =  torch.tensor(0, dtype=torch.float32).to(device)
        for fm_idx in range(len(preds)):
            for ar_idx in range(len(preds[fm_idx])):
                gt_idxs = gt_cats[fm_idx][ar_idx] != 20
                loss.add_(
                    self._bbs_loss(gt_bbs[fm_idx][ar_idx], preds[fm_idx][ar_idx][0], gt_idxs))
        return loss

    def _bbs_loss(self, y, yhat, gt_idxs):
        batch_size = y.shape[0]
        y = torch.tensor(y.reshape(batch_size, -1, 4)[gt_idxs], dtype=torch.float32).to(device)
        inputs = yhat.reshape(batch_size, -1, 4)[gt_idxs]
        targets = (y / SIZE)
        return F.smooth_l1_loss(inputs, targets, reduction='sum')


class SSDLoss(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.bbs_loss = BbsL1Loss()
        self.cats_loss = CatsBCELoss()

    def forward(self, inputs, targets):
        preds = inputs
        gt_bbs, gt_cats = targets
        conf = self.cats_loss(preds, gt_cats)
        loc = self.bbs_loss(preds, (gt_bbs, gt_cats))
        n = self._matched_gt_cats(gt_cats)
        return (1/n) * (conf + (self.alpha*loc))

    def _matched_gt_cats(self, gt_cats):
        n = torch.tensor(0, dtype=torch.float32).to(device)
        for fm_idx in range(len(gt_cats)):
            for ar_idx in range(len(gt_cats[fm_idx])):
                gt_idxs = gt_cats[fm_idx][ar_idx] != 20
                n += gt_idxs.sum().type(torch.float32).to(device)
        return n
