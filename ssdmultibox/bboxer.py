"""
Module for bounding box (bbs) operations
"""
import numpy as np
import torch

from ssdmultibox.config import cfg


class Bboxer:
    """
    Stores most base bounding box operations and 6x6 /
    feature_map x aspect_ratio operations
    """

    def __init__(self):
        raise AssertionError("this class only supports class methods")

    @classmethod
    def anchor_centers(cls, grid_size):
        "Returns the x,y center coordinates for all anchor boxes"
        anc_offset = 1/(grid_size*2)
        anc_x = np.repeat(np.linspace(anc_offset, 1-anc_offset, grid_size), grid_size)
        anc_y = np.tile(np.linspace(anc_offset, 1-anc_offset, grid_size), grid_size)
        return np.stack([anc_x,anc_y], axis=1)

    @classmethod
    def anchor_sizes(cls, grid_size, aspect_ratio=(1.,1.)):
        "Returns a 2d arr of the archor sizes per the aspect_ratio"
        sk = 1/grid_size
        w = sk * aspect_ratio[0]
        h = sk * aspect_ratio[1]
        return np.reshape(np.repeat([w, h], grid_size*grid_size), (2,-1)).T

    @classmethod
    def anchor_corners(cls, grid_size, aspect_ratio=(1.,1.)):
        "Return 2d arr where each item is [x1,y1,x2,y2] of top left and bottom right corners"
        centers = cls.anchor_centers(grid_size)
        hw = cls.anchor_sizes(grid_size, aspect_ratio)
        return cls.hw2corners(centers, hw)

    @classmethod
    def anchor_boxes(cls, feature_maps=None, aspect_ratios=None):
        """
        Return 6x6 all feature_map by aspect_ratio anchor boxes

        Args:
            feature_maps (1d array): of integer feature map sizes
            aspect_ratios (function):
                that returns an aspect ratios array, fallows signature
                `aspect_ratios(grid_size=1)`
        """
        feature_maps = feature_maps or cfg.FEATURE_MAPS
        aspect_ratios = aspect_ratios or cls.aspect_ratios
        boxes = []
        for i in range(len(feature_maps)):
            grid_size = feature_maps[i]
            ar_bbs = []
            for aspect_ratio in aspect_ratios(grid_size):
                ar_bbs.append(
                    cls.anchor_corners(grid_size, aspect_ratio)
                )
            boxes.append(ar_bbs)
        return boxes

    @classmethod
    def hw2corners(cls, center, hw):
        "Return anchor corners based upon center and hw"
        return np.concatenate([center-hw/2, center+hw/2], axis=1)

    @staticmethod
    def aspect_ratios(grid_size=1):
        "Returns the aspect ratio"
        sk = 1. / grid_size
        return np.array([
            (1., 1.),
            (2., 1.),
            (3., 1.),
            (1., 2.),
            (1., 3.),
            (np.sqrt(sk*sk+1), 1.)
        ])

    @classmethod
    def get_intersection(cls, bbs, im, grid_size=4, aspect_ratio=(1.,1.)):
        # returns the i part of IoU scaled [0,1]
        bbs = cls.scaled_fastai_bbs(bbs, im)
        bbs_count = grid_size*grid_size
        bbs16 = np.reshape(np.tile(bbs, bbs_count), (-1,bbs_count,4))
        anchor_corners = cls.anchor_corners(grid_size, aspect_ratio)
        intersect = np.minimum(
            np.maximum(anchor_corners[:,:2], bbs16[:,:,:2]) - \
            np.minimum(anchor_corners[:,2:], bbs16[:,:,2:]), 0)
        return intersect[:,:,0] * intersect[:,:,1]

    @staticmethod
    def scaled_fastai_bbs(bbs, im):
        """
        Args:
            bbs (list): pascal bb of [x, y, abs_x-x, abs_y-y]
            im (np.array): 3d HWC
        Returns:
            (np.array): fastai bb of [y, x, abs_y, abs_x]
        """
        im_w = im.shape[1]
        im_h = im.shape[0]
        bbs = np.divide(bbs, [im_w, im_h, im_w, im_h])
        return np.array([
            bbs[:,1],
            bbs[:,0],
            bbs[:,3]+bbs[:,1]-(1/cfg.SIZE),
            bbs[:,2]+bbs[:,0]-(1/cfg.SIZE)]).T

    @classmethod
    def get_ancb_area(cls, grid_size):
        "Returns the [0,1] normalized area of a single anchor box"
        return 1. / np.square(grid_size)

    @classmethod
    def get_bbs_area(cls, bbs, im):
        "Returns a np.array of the [0,1] normalized bbs area"
        bbs = cls.scaled_fastai_bbs(bbs, im)
        return np.abs(bbs[:,0]-bbs[:,2])*np.abs(bbs[:,1]-bbs[:,3])

    @classmethod
    def get_iou(cls, bbs, im, grid_size=4, aspect_ratio=(1.,1.)):
        "Returns a 2d arr of the IoU for each obj with size [obj count, feature cell count]"
        intersect = cls.get_intersection(bbs, im, grid_size, aspect_ratio)
        bbs_union = cls.get_ancb_area(grid_size) + cls.get_bbs_area(bbs, im)
        bbs_union_all = np.repeat(
            bbs_union, intersect.shape[-1]).reshape(*intersect.shape) - intersect
        return intersect / bbs_union_all

    @classmethod
    def get_gt_overlap_and_idx(cls, bbs, im, grid_size=4, aspect_ratio=(1.,1.)):
        "Returns a 1d arr for all feature cells with the gt overlaps and gt_idx"
        overlaps = torch.tensor(cls.get_iou(bbs, im, grid_size, aspect_ratio))
        _, prior_idx = overlaps.max(1)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        # sets the gt_idx equal to the obj_idx for each prior_idx
        for i,o in enumerate(prior_idx):
            gt_idx[o] = i
        return gt_overlap, gt_idx

    @classmethod
    def get_gt_bbs_and_cats_for_all(cls, bbs, cats, im):
        """
        Returns all gt_cats and gt_bbs for all grid sizes and aspect ratios

        Dimensions are as follows. Each item is (bbs, cats):
        ar - aspect ratio
        fm - feature map
        [ [[fm1 ar1], [fm1 ar2], ...],
          [[fm2 ar1], [fm2 ar2], ...]]
        """
        all_bbs = []
        all_cats = []
        for grid_size in cfg.FEATURE_MAPS:
            ar_bbs = []
            ar_cats = []
            for aspect_ratio in cls.aspect_ratios(grid_size):
                gt_bbs, gt_cats = cls.get_gt_bbs_and_cats(bbs, cats, im, grid_size, aspect_ratio)
                # populate list(s)
                ar_bbs.append(gt_bbs)
                ar_cats.append(gt_cats)
            all_bbs.append(ar_bbs)
            all_cats.append(ar_cats)
        return all_bbs, all_cats

    @classmethod
    def get_gt_bbs_and_cats(cls, bbs, cats, im, grid_size=4, aspect_ratio=(1.,1.)):
        """
        Returns bbs per anchor box gt labels and dense gt_cats

        Args:
            bbs (2d list): of pascal bbs int coordinates
            cats (2d list):
                sparse list of int categories, one-hot encoded with the final
                'bg' category shaved off
                NOTE: nuance here, this allows the model to predict that there
                are no objects identified, but it's behavior isn't guided
                towards predicting the 'bg' everytime just to naively minimize
                the cost
            im (2d list): of HWC raw pascal im
            grid_size (int): feature map dimen
            aspect_ratio (tuple): w,h aspect ratio
        Returns:
            bbs (1d list) - every 4 items matches the cat, cats (1d list)
        """
        gt_overlap, gt_idx = cls.get_gt_overlap_and_idx(bbs, im, grid_size, aspect_ratio)
        bbs = np.multiply(cls.scaled_fastai_bbs(bbs, im), cfg.SIZE)
        cats = np.array(cats)
        gt_bbs = bbs[gt_idx]
        gt_cats = cats[gt_idx]
        # the previous line will set the 0 ith class as the gt, so
        # set it to bg if it doesn't meet the IoU threshold
        pos = gt_overlap > cfg.IOU_THRESH
        try:
            neg_idx = np.nonzero(1-pos)[:,0]
        except IndexError:
            # skip if no negative indexes
            pass
        else:
            gt_cats[neg_idx] = 20

        return np.reshape(gt_bbs, (-1)), gt_cats

    @staticmethod
    def one_hot_encode(gt_cats, num_classes):
        """
        Returns cats one-hot encoded

        Args:
            gt_cats (1d list): dense vector of category labels
            num_classes (int): number of posible category classes
        Returns:
            (2d list): shape = (len(gt_cats), num_classes)
        """
        return np.eye(num_classes)[gt_cats]

    @classmethod
    def one_hot_encode_no_bg(cls, gt_cats, num_classes):
        """
        Returns the one-hot encoded cats with the bg cat sliced off
        because we want to disregard the bg cat in the loss func
        """
        return cls.one_hot_encode(gt_cats, num_classes)[:,:-1]

    @staticmethod
    def pascal_bbs(bbs):
        """
        Returns fastai_bbs as pascal formatted bbs

        Args:
            bbs (2d list): fastai encoded bbs
        """
        return np.array([bbs[:,1], bbs[:,0], bbs[:,3]-bbs[:,1]+1, bbs[:,2]-bbs[:,0]+1]).T

    @staticmethod
    def scaled_pascal_bbs(bbs, im, scale=1):
        "Returns scaled pascal bbs scaled [0,1]"
        pascal_bbs = np.array(bbs)
        im_w = im.shape[1]
        im_h = im.shape[0]
        return np.divide(pascal_bbs, [im_w, im_h, im_w, im_h]) * scale

    @staticmethod
    def fastai_bb_to_pascal_bb(bb):
        "Converts a fastai formatted bb to a pascal bb"
        return np.array([bb[1], bb[0], bb[3]-bb[1], bb[2]-bb[0]])

    # REVIEW: might need 2 separate classes for the Bboxer concept
    # one that deals w/ single bbox ops and one for multiple

    @staticmethod
    def single_bb_intersect(a, b):
        "Returns the area of the intersection of 2 bb"
        wh = np.minimum(
            np.maximum(a[:2], b[:2]) - np.minimum(a[2:], b[2:]), 0)
        return wh[0] * wh[1]

    @staticmethod
    def single_bb_area(bb):
        "Returns the bb area"
        return np.abs(bb[0]-bb[2])*np.abs(bb[1]-bb[3])

    @classmethod
    def single_bb_iou(cls, bb, gt_bb):
        i = cls.single_bb_intersect(bb, gt_bb)
        # don't forget to remove their overlapping area from the union calc!
        u = cls.single_bb_area(bb) + cls.single_bb_area(gt_bb) - i
        return i/u

    # stacked - in the Bboxer class for now..

    @classmethod
    def get_stacked_anchor_boxes(cls, feature_maps=None, aspect_ratios=None):
        """
        Return stacked anchor bbs coordinates for all feature_map
        by aspect_ratio anchor boxes. For anchor bbs, the coordinates
        are [x, y, x2, y2]

        Args:
            feature_maps (1d array): of integer feature map sizes
            aspect_ratios (function):
                that returns an aspect ratios array, fallows signature
                `aspect_ratios(grid_size=1)`
        Returns:
            (2d array): of shape (anchor_bbs_count, 4)
        """
        feature_maps = feature_maps or cfg.FEATURE_MAPS
        aspect_ratios = aspect_ratios or cls.aspect_ratios
        boxes = None
        for i, fm in enumerate(feature_maps):
            grid_size = fm
            for aspect_ratio in aspect_ratios(grid_size):
                arr = np.clip(cls.anchor_corners(grid_size, aspect_ratio), 0, 1)
                if not isinstance(boxes, np.ndarray):
                    boxes = arr
                else:
                    boxes = np.concatenate((boxes, arr))
        return boxes

    @classmethod
    def get_stacked_intersection(cls, bbs, im, stacked_anchor_boxes):
        """
        Returns stacked intersections of shape (bbs_count, anchor_bbs_count)

        Args:
            bbs (2d array): from annotations of raw gt bbs
            im (HWC 3d array) of image
            stacked_anchor_boxes (2d array):
                of shape (n, 4) where n is the number of stacked anchor boxes
        """
        bbs = cls.scaled_fastai_bbs(bbs, im)
        stacked_anchor_boxes_count = stacked_anchor_boxes.shape[0]
        bbs16 = np.reshape(
            np.tile(bbs, stacked_anchor_boxes_count), (-1,stacked_anchor_boxes_count,4))
        intersect = np.minimum(
            np.maximum(stacked_anchor_boxes[:,:2], bbs16[:,:,:2]) - \
            np.minimum(stacked_anchor_boxes[:,2:], bbs16[:,:,2:]), 0)
        return intersect[:,:,0] * intersect[:,:,1]

    @classmethod
    def get_anchor_box_area(cls, sab):
        """
        Returns the area of each stacked_anchor_box in a 1d array

        Args:
            sab (2d array): stacked_anchor_boxes
        """
        return (sab[:,2]-sab[:,0])*(sab[:,3]-sab[:,1])

    @classmethod
    def get_stacked_union(cls, stacked_anchor_boxes, stacked_intersect, bbs_area):
        """
        Returns the stacked unioned area of the intersect area and the
        anchor box area

        Args:
            stacked_anchor_boxes (2d array): of shape (anchor_bbs_count, bbs_count)
            stacked_intersect (2d array): of shape (bbs_count, anchor_bbs_count)
            bbs_area (1d array): of shape (bbs_count,)
        Returns:
            2d array: of shape (bbs_count, anchor_bbs_count)
        """
        anc_bbs_area = cls.get_anchor_box_area(stacked_anchor_boxes)
        bbs_area = np.repeat(bbs_area, stacked_intersect.shape[1]).reshape(
            stacked_intersect.shape[0], -1)
        return anc_bbs_area + bbs_area - stacked_intersect

    @classmethod
    def get_stacked_iou(cls, stacked_anchor_boxes, stacked_intersect, bbs_area):
        """
        Returns the stacked IoU

        Args:
            stacked_anchor_boxes (2d array): of shape (anchor_bbs_count, bbs_count)
            stacked_intersect (2d array): of shape (bbs_count, anchor_bbs_count)
            bbs_area (1d array): of shape (bbs_count,)
        Returns:
            2d array: of shape (bbs_count, anchor_bbs_count)
        """
        union = cls.get_stacked_union(stacked_anchor_boxes, stacked_intersect, bbs_area)
        return stacked_intersect / union

    @classmethod
    def get_stacked_gt_overlap_and_idx(cls, stacked_anchor_boxes, stacked_intersect, bbs_area):
        """
        Returns the gt_overlap and gt_idx for the bbs based on the IoU. The ground
        truth overlap and idx is the object with the highest IoU per object.

        NOTE: it is possible for a tie, in which the last tied object wins

        Args:
            stacked_anchor_boxes (2d array): of shape (anchor_bbs_count, bbs_count)
            stacked_intersect (2d array): of shape (bbs_count, anchor_bbs_count)
            bbs_area (1d array): of shape (bbs_count,)
        Returns:
            2 item tuple (1d array of float overlaps, 1d array of int object idxs)
            Each is the length of the number of anchor boxes
        """
        overlaps = torch.tensor(
            cls.get_stacked_iou(stacked_anchor_boxes, stacked_intersect, bbs_area))
        _, prior_idx = overlaps.max(1)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        # sets the gt_idx equal to the obj_idx for each prior_idx
        for i,o in enumerate(prior_idx):
            gt_idx[o] = i
        return gt_overlap, gt_idx

    @classmethod
    def get_stacked_gt_bbs_and_cats(
            cls, bbs, cats, im, stacked_anchor_boxes, stacked_intersect, bbs_area):
        """
        Returns the gt_bbs and gt_cats based upon the gt_overlaps or if the
        IoU is greater than the cfg.IOU_THRESH hyperparameter

        Args:
            bbs (2d array): of raw bbs
            cats (1d array): of category ids
            im (3d array): raw image array
            stacked_anchor_boxes (2d array): of shape (anchor_bbs_count, bbs_count)
            stacked_intersect (2d array): of shape (bbs_count, anchor_bbs_count)
            bbs_area (1d array): of shape (bbs_count,)
        Returns:
            2 item tuple (2d array of bbs, 1d array of cat ids)
        """
        gt_overlap, gt_idx = cls.get_stacked_gt_overlap_and_idx(
            stacked_anchor_boxes, stacked_intersect, bbs_area)

        scaled_bbs = cls.scaled_fastai_bbs(bbs, im)
        gt_bbs = scaled_bbs[gt_idx]
        gt_cats = cats[gt_idx]
        # the previous line will set the 0 ith class as the gt, so
        # set it to bg if it doesn't meet the IoU threshold
        pos = gt_overlap > cfg.IOU_THRESH
        try:
            neg_idx = np.nonzero(1-pos)[:,0]
        except IndexError:
            # skip if no negative indexes
            pass
        else:
            gt_cats[neg_idx] = 20

        return gt_bbs, gt_cats

    @classmethod
    def get_stacked_gt(cls, bbs, cats, im, feature_maps=None, aspect_ratios=None):
        """
        Wrapper method for getting the gt_bbs and gt_cats based upon what
        data the Dataset returns

        Args:
            bbs (2d array): of raw bbs
            cats (1d array): of category ids
            im (3d array): raw image array
            feature_maps (1d array): of integer feature map sizes
            aspect_ratios (function):
                that returns an aspect ratios array, fallows signature
                `aspect_ratios(grid_size=1)`
        Returns:
            2 item tuple (2d array of bbs, 1d array of cat ids)
        """
        feature_maps = feature_maps or cfg.FEATURE_MAPS
        aspect_ratios = aspect_ratios or cls.aspect_ratios
        scaled_bbs = cls.scaled_fastai_bbs(bbs, im)
        stacked_anchor_boxes = cls.get_stacked_anchor_boxes(feature_maps, aspect_ratios)
        stacked_intersect = cls.get_stacked_intersection(bbs, im, stacked_anchor_boxes)
        bbs_area = cls.get_bbs_area(bbs, im)
        return cls.get_stacked_gt_bbs_and_cats(
            bbs, cats, im, stacked_anchor_boxes, stacked_intersect, bbs_area)


class TensorBboxer(Bboxer):

    @classmethod
    def get_stacked_anchor_boxes(cls, feature_maps=None, aspect_ratios=None):
        feature_maps = feature_maps or cfg.FEATURE_MAPS
        bbs = super().get_stacked_anchor_boxes(feature_maps, aspect_ratios)
        return torch.tensor(bbs, dtype=torch.float32).to(cfg.DEVICE)

    @staticmethod
    def get_allowed_offset(cell_size):
        """
        Returns the max offset allowed per the cell_size

        Args:
            cell_size (int): should be a cfg.FEATURE_MAPS int cell size
        Returns:
            float
        """
        return (1/cell_size) * cfg.ALLOWED_OFFSET

    @classmethod
    def get_feature_map_max_offsets(cls):
        """
        Returns a 2d tensor fo max offsets allowed per the feature
        map cell sizes. Shape is (feature_maps_count aka 11640, 1)
        """
        return torch.cat(
            [torch.FloatTensor(fm*fm*cfg.ASPECT_RATIOS,1).fill_(cls.get_allowed_offset(fm))
             for fm in cfg.FEATURE_MAPS],
        dim=0).to(cfg.DEVICE)
