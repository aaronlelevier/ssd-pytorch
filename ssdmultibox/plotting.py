import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
from ssdmultibox.datasets import SIZE, Bboxer
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
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


def plot_single(dataset, idx, ax=None):
    "uses the Dataset idx to select a training sample and plot it"
    image_id, im, gt_bbs, gt_cats = dataset[idx]
    gt_idx = np.where(gt_cats == 1)[0]
    gt_idx_bbs = gt_bbs[gt_idx]
    gt_idx_cats = np.argmax(gt_cats, axis=1)[gt_idx]

    resized_image = np.transpose(im, (1,2,0))
    ax = show_img(resized_image, ax=ax)
    for bbox in dataset.bboxer.anchor_corners(grid_size=4):
        draw_rect(ax, bbox*224, edgecolor='red')
    for bbox, cat in zip(dataset.bboxer.pascal_bbs(gt_idx_bbs), gt_idx_cats):
        draw_rect(ax, bbox)
        draw_text(ax, bbox[:2], dataset.categories()[cat])


def plot_multiple(dataset):
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i,ax in enumerate(axes.flat):
        plot_single(dataset, i, ax)
    plt.tight_layout()


def plot_single_nms_detections(dataset, idx, detections, limit=5):
    """
    Shows NMS detections of an image

    Args:
        dataset (torch.utils.data.Dataset)
        idx (int): index of dataset item to show
        detections (3 item tuple): return value from `Predict.predict_all`
        limit (int:optional): use to limit the number of detections shown
    """
    image_id, im, gt_bbs, gt_cats = dataset[idx]
    ann = dataset.get_annotations()[image_id]
    # image
    im = open_image(ann['image_path'])
    resized_im = cv2.resize(im, (SIZE, SIZE))
    ax = show_img(resized_im)
    # detections
    if detections:
        detected_bbs, scores, ids = detections
        for i, (bb, score, cls_id) in enumerate(zip(detected_bbs, scores, ids)):
            pascal_bb = Bboxer.fastai_bb_to_pascal_bb(bb*SIZE)
            draw_rect(ax, pascal_bb)
            draw_text(ax, pascal_bb[:2], f'{score.item()}'[:6], sz=9)
            if i == limit-1:
                break
    # show gt bbs
    for gt_bb in (Bboxer.scaled_pascal_bbs(np.array(ann['bbs']), im) * SIZE):
        draw_rect(ax, gt_bb, edgecolor='yellow')


def plot_single_predictions(dataset, idx, targets):
    """
    Plots the gt bb(s) and predicted bbs

    Args:
        dataset (torch.utils.data.Dataset)
        idx (int): index of dataset item to show
        targets (2d array):
            fastai formatted bbs to plot
    """
    image_id, im, gt_bbs, gt_cats = dataset[idx]
    ann = dataset.get_annotations()[image_id]
    im = open_image(ann['image_path'])
    resized_im = cv2.resize(im, (SIZE, SIZE))
    ax = show_img(resized_im)

    for gt_bb in (Bboxer.scaled_pascal_bbs(np.array(ann['bbs']), im) * SIZE):
        draw_rect(ax, gt_bb, edgecolor='yellow')

    for i, bb in enumerate(targets):
        gt_overlap_bb = Bboxer.fastai_bb_to_pascal_bb(bb)* SIZE
        draw_rect(ax, gt_overlap_bb, edgecolor='red')
        draw_text(ax, gt_overlap_bb[:2], i, sz=8)


def get_anchor_bbs_targets(gt_cats, idx):
    """
    Returns a 2d array of the target fastai formatted bbs
    based upon the gt_cats that aren't background

    Args:
        gt_cats (3d array): of batch gt_cats
        idx (int): gt_cats item from the batch to retrieve
    """
    gt_cat = torch.tensor(gt_cats[idx])
    not_bg_mask = gt_cat != 20
    not_bg_mask = (not_bg_mask == 1).nonzero()
    not_bg_mask = not_bg_mask.squeeze(1)
    stacked_anchor_boxes = torch.tensor(
        Bboxer.get_stacked_anchor_boxes(), dtype=preds.dtype).to(device)
    return stacked_anchor_boxes[not_bg_mask]
