import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, patheffects
from ssdmultibox.utils import open_image
from ssdmultibox.datasets import Bboxer, SIZE

import cv2

def show_img(im, figsize=None, ax=None):
    if not ax:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def draw_rect(ax, b, edgecolor='white'):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=edgecolor, lw=2))
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


def plot_single_detections(dataset, ids, detections):
    image_id, im, gt_bbs, gt_cats = dataset[idx]
    ann = dataset.get_annotations()[image_id]
    # image
    im = open_image(ann['image_path'])
    resized_im = cv2.resize(im, (SIZE, SIZE))
    ax = show_img(resized_im)
    # detections
    if detections:
        detected_bbs, _ = detections
        for bb in detected_bbs:
            draw_rect(ax, Bboxer.fastai_bb_to_pascal_bb(bb*SIZE))
