import torch
import torch.nn.functional as F
from torch import nn

from ssdmultibox.datasets import NUM_CLASSES, SIZE, device


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
        if len(y.shape) == 1:
            cats_preds = yhat.reshape(batch_size, NUM_CLASSES)[:,:-1]
        else:
            cats_preds = yhat.reshape(batch_size, -1, NUM_CLASSES)[:,:,:-1]
        gt_idxs = y != 20
        return F.cross_entropy(
            cats_preds[gt_idxs], y[gt_idxs].type(torch.long), reduction='sum')

    # NOTE: not in use ...
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
        conf = self.cats_loss(preds, gt_cats).abs()
        loc = self.bbs_loss(preds, (gt_bbs, gt_cats)).abs()
        n = self._matched_gt_cats(gt_cats)
        print('cats_loss:', conf.item())
        print('bbs_loss:', loc.item())
        return (1/n) * (conf + (self.alpha*loc))

    def _matched_gt_cats(self, gt_cats):
        n = torch.tensor(0, dtype=torch.float32).to(device)
        for fm_idx in range(len(gt_cats)):
            for ar_idx in range(len(gt_cats[fm_idx])):
                gt_idxs = gt_cats[fm_idx][ar_idx] != 20
                n += gt_idxs.sum().type(torch.float32).to(device)
        return n
