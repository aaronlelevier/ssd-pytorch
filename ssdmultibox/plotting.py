import numpy as np
from matplotlib import patches, patheffects
from matplotlib import pyplot as plt

from ssdmultibox.bboxer import Bboxer, TensorBboxer
from ssdmultibox.config import cfg
from ssdmultibox.predict import Predict


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


def get_anchor_bbs_targets(idx, gt_cats):
    """
    Returns the anchor bbs and cats as labeled by the max IoU and thresh rules
    """
    mask = gt_cats[idx] != 20
    anchor_boxes = TensorBboxer.get_stacked_anchor_boxes()
    bbs = anchor_boxes[mask]
    cats = gt_cats[idx][mask]
    return bbs, cats


def get_pred_targets(idx, gt_cats, preds):
    """
    Return the pred bbs and cats as labeled by the max IoU and thresh rules
    """
    mask = gt_cats[idx] != 20
    bbs_preds, cats_preds = preds
    bbs = bbs_preds[idx][mask]
    # add [:,:-1] so we don't predict bg!
    _, cats = cats_preds[idx][mask][:,:-1].max(1)
    return bbs, cats


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


def plot_single_predictions(
        chw_im, gt_bbs, gt_cats, idx, cat_names, targets, ax=None):
    im = np.transpose(chw_im, (1,2,0))
    ax = show_img(im, ax=ax)

    mask = gt_cats[idx] != 20
    # anchors
    for bb in gt_bbs[idx][mask]:
        pascal_bb = Bboxer.fastai_bb_to_pascal_bb(bb) * cfg.NORMALIZED_SIZE
        draw_rect(ax, pascal_bb, edgecolor='yellow')

    for bb, cat in zip(*targets):
        pascal_bb = Bboxer.fastai_bb_to_pascal_bb(bb) * cfg.NORMALIZED_SIZE
        draw_rect(ax, pascal_bb, edgecolor='red')
        draw_text(ax, pascal_bb[:2], cat_names[cat.item()], sz=8)


def plot_anchor_bbs(model_output, idx, dataset, ax=None):
    cat_names = dataset.categories()
    image_ids, ims, gt_bbs, gt_cats = model_output
    chw_im = ims[idx]
    plot_single_predictions(
        chw_im, gt_bbs, gt_cats, idx, cat_names,
        targets=get_anchor_bbs_targets(idx, gt_cats), ax=ax)


def plot_preds(model_output, idx, dataset, preds, ax=None):
    cat_names = dataset.categories()
    image_ids, ims, gt_bbs, gt_cats = model_output
    chw_im = ims[idx]
    plot_single_predictions(
        chw_im, gt_bbs, gt_cats, idx, cat_names,
        targets=get_pred_targets(idx, gt_cats, preds), ax=ax)


def plot_nms_preds(model_output, idx, dataset, preds, limit=5, ax=None):
    cat_names = dataset.categories()
    image_ids, ims, gt_bbs, gt_cats = model_output
    chw_im = ims[idx]
    boxes, scores, cls_ids = Predict.all(preds, index=idx)
    plot_single_predictions(
        chw_im, gt_bbs, gt_cats, idx, cat_names,
        targets=(boxes[:limit], cls_ids[:limit]), ax=ax)


def plot_nms_single_preds(model_output, idx, dataset, preds, cls_id, limit=5, ax=None):
    "Plots NMS predictions for a single object class"
    cat_names = dataset.categories()
    image_ids, ims, gt_bbs, gt_cats = model_output
    chw_im = ims[idx]
    nms_preds = Predict.single(cls_id, preds, index=idx)
    # guard agains no NMS preds being found based upon the confidence threshold
    if nms_preds:
        boxes, scores, cls_ids = nms_preds
        targets = (boxes[:limit], cls_ids[:limit])
    else:
        targets = ([], [])
    plot_single_predictions(
        chw_im, gt_bbs, gt_cats, idx, cat_names, targets=targets, ax=ax)
