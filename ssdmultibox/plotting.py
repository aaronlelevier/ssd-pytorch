import cv2
import numpy as np
import torch
from matplotlib import patches, patheffects
from matplotlib import pyplot as plt

from ssdmultibox.bboxer import Bboxer, TensorBboxer
from ssdmultibox.config import cfg
from ssdmultibox.predict import Predict
from ssdmultibox.utils import open_image


def show_img(im, figsize=None, ax=None):
    if not ax:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def draw_rect(ax, pascal_bb, edgecolor='white'):
    patch = ax.add_patch(patches.Rectangle(
        pascal_bb[:2], *pascal_bb[-2:], fill=False, edgecolor=edgecolor, lw=2))
    draw_outline(patch, 4)


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(
        *xy, txt, verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


def plot_single_predictions(dataset, idx, targets, ax=None):
    """
    Plots the gt bb(s) and predicted bbs

    Args:
        dataset (torch.utils.data.Dataset)
        idx (int): index of dataset item to show
        targets (2d array):
            fastai formatted bbs to plot, [0,1] normalized
    """
    image_id, im, *_ = dataset[idx]
    ann = dataset.get_annotations()[image_id]
    im = open_image(ann['image_path'])
    resized_im = cv2.resize(im, (cfg.SIZE, cfg.SIZE))
    ax = show_img(resized_im, ax=ax)

    for gt_bb in Bboxer.scaled_pascal_bbs(np.array(ann['bbs']), im, scale=cfg.SIZE):
        draw_rect(ax, gt_bb, edgecolor='yellow')

    cat_names = dataset.categories()
    for bb, cat in zip(*targets):
        pascal_bb = Bboxer.fastai_bb_to_pascal_bb(bb)
        draw_rect(ax, pascal_bb, edgecolor='red')
        draw_text(ax, pascal_bb[:2], cat_names[cat.item()], sz=8)


def get_targets(gt_cats, idx, bbs_preds=None):
    """
    Returns target bbs,cats of either the anchor boxes or the bbs_preds
    using the anchor box offsets
    """
    gt_cat = torch.tensor(gt_cats[idx])
    not_bg_mask = gt_cat != 20
    not_bg_mask = (not_bg_mask == 1).nonzero()
    not_bg_mask = not_bg_mask.squeeze(1)
    cats = gt_cats[idx][not_bg_mask]
    if isinstance(bbs_preds, torch.Tensor):
        bbs = bbs_preds[idx][not_bg_mask]
    else:
        anchor_boxes = TensorBboxer.get_stacked_anchor_boxes()
        bbs = anchor_boxes[not_bg_mask] * cfg.SIZE
    return bbs, cats


def plot_anchor_bbs(dataset, image_ids, idx, gt_cats, ax=None):
    "Plots the ground truth anchor boxes"
    image_id = image_ids[idx].item()
    dataset_idx = dataset.get_image_id_idx_map()[image_id]
    plot_single_predictions(
        dataset, dataset_idx,
        targets=get_targets(gt_cats, idx), ax=ax)


def plot_preds(dataset, image_ids, idx, bbs_preds, gt_cats, ax=None):
    "Plots the predictions based on the ground truth anchor box offsets"
    image_id = image_ids[idx].item()
    dataset_idx = dataset.get_image_id_idx_map()[image_id]
    plot_single_predictions(
        dataset, dataset_idx,
        targets=get_targets(gt_cats, idx, bbs_preds), ax=ax)


def plot_nms_preds(dataset, image_ids, idx, preds, limit=5, ax=None):
    "Plots NMS predictions"
    image_id = image_ids[idx].item()
    dataset_idx = dataset.get_image_id_idx_map()[image_id]
    boxes, scores, cls_ids = Predict.all(preds, index=idx)
    plot_single_predictions(
        dataset, dataset_idx, targets=(boxes[:limit], cls_ids[:limit]), ax=ax)


def plot_nms_single_preds(dataset, image_ids, idx, cls_id, preds, limit=5, ax=None):
    "Plots NMS predictions for a single object class"
    image_id = image_ids[idx].item()
    dataset_idx = dataset.get_image_id_idx_map()[image_id]
    boxes, scores, cls_ids = Predict.single(cls_id, preds, index=idx)
    plot_single_predictions(
        dataset, dataset_idx, targets=(boxes[:limit], cls_ids[:limit]), ax=ax)


def plot_multiple(func, plots=(2,2), **kwargs):
    """
    Plot multiple training examples using the above plotting functions

    Args:
        func (plotting func)
        plots (tuple):
            this is the N x M size of the plot. It should match the
            number of samples in the batch
        kwargs: plotting func specific arguments
    """
    _, axes = plt.subplots(*plots, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        func(idx=i, ax=ax, **kwargs)
    plt.tight_layout()
