import torch
import torch.nn.functional as F

from ssdmultibox.datasets import NUM_CLASSES


def cats_loss(y, yhat):
    try:
        cats_label = torch.eye(NUM_CLASSES)[y][:,:,:-1]
    except IndexError:
        # final preds only have one feature_map cell, so this is needed
        cats_label = torch.eye(NUM_CLASSES)[y][:,:-1].reshape(4, -1, 20)

    cats_preds = yhat.reshape(4, -1, NUM_CLASSES)[:,:,:-1]
    gt_idxs = y != 20
    return F.binary_cross_entropy_with_logits(cats_label[gt_idxs], cats_preds[gt_idxs])


def all_cat_loss(gt_cats, preds):
    loss =  torch.tensor(0, dtype=torch.float32)

    for fm_idx in range(len(preds)):
        for ar_idx in range(len(preds[fm_idx])):
            loss.add_(
                cats_loss(gt_cats[fm_idx][ar_idx], preds[fm_idx][ar_idx][1]))
    return loss


def bbs_loss(y, yhat, gt_idxs):
    y = torch.tensor(y.reshape(4, -1, 4)[gt_idxs], dtype=torch.float32)
    return (((y / SIZE) - yhat.reshape(4, -1, 4)[gt_idxs]).abs()).mean()


def all_bbs_loss(gt_bbs, preds, gt_cats):
    loss =  torch.tensor(0, dtype=torch.float32)

    for fm_idx in range(len(preds)):
        for ar_idx in range(len(preds[fm_idx])):
            gt_idxs = gt_cats[fm_idx][ar_idx] != 20
            loss.add_(
                bbs_loss(gt_bbs[fm_idx][ar_idx], preds[fm_idx][ar_idx][0], gt_idxs))
    return loss
