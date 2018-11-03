import torch

from ssdmultibox.datasets import NUM_CLASSES


def cats_loss(y, yhat):
    try:
        cats_label = torch.eye(NUM_CLASSES)[y][:,:,:-1]
    except IndexError:
        # final preds only have one feature_map cell, so this is needed
        cats_label = torch.eye(NUM_CLASSES)[y][:,:-1].reshape(4, -1, 20)

    cats_preds = yhat.reshape(4, -1, NUM_CLASSES)[:,:,:-1]
    gt_idxs = y != 20
    return ((cats_preds[gt_idxs] - cats_label[gt_idxs]).abs()).mean()


def all_cat_loss(gt_cats, preds):
    cat_loss =  torch.tensor(0, dtype=torch.float32)

    for fm_idx in range(len(preds)):
        for ar_idx in range(len(preds[fm_idx])):
            cat_loss.add_(
                cats_loss(gt_cats[fm_idx][ar_idx], preds[fm_idx][ar_idx][1]))
    return cat_loss
