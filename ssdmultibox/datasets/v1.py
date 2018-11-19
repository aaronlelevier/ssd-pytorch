import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ssdmultibox import config
from ssdmultibox.utils import open_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SIZE = 300

NUM_CLASSES = 21

BATCH = 4

# IoU threshold
THRESH = 0.5

FEATURE_MAPS = [38, 19, 10, 5, 3, 1]

IMAGES = 'images'
ANNOTATIONS = 'annotations'
CATEGORIES = 'categories'
ID = 'id'
NAME = 'name'
IMAGE_ID = 'image_id'
BBOX = 'bbox'
BBS = 'bbs'
CATS = 'cats'
CATEGORY_ID = 'category_id'
FILE_NAME = 'file_name'
IMAGE = 'image'
CATEGORY = 'category'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
IMAGE = 'image'
IMAGE_PATH = 'image_path'


class PascalDataset(Dataset):

    def __init__(self):
        self.filepath = config.DATA_DIR
        self.bboxer = Bboxer

    @property
    def pascal_json(self):
        "Returns the json data per the mode. i.e. train, val, test"
        raise NotImplementedError('pascal_json')

    def __len__(self):
        return len(self.raw_images())

    def __getitem__(self, idx):
        image_ids = self.get_image_ids()
        image_id = image_ids[idx]
        ann = self.get_annotations()[image_id]
        bbs = ann[BBS]
        cats = ann[CATS]

        image_paths = self.images()
        im = open_image(image_paths[image_id])
        chw_im = self.scaled_im_by_size_and_chw_format(im)

        gt_bbs, gt_cats = self.bboxer.get_gt_bbs_and_cats_for_all(bbs, cats, im)

        return image_id, chw_im, gt_bbs, gt_cats

    @staticmethod
    def to_device(ims, gt_bbs, gt_cats):
        """
        Utility function to put the dataset outputs on the correct device
        """
        # put data on device
        ims = torch.tensor(ims, dtype=torch.float32).to(device)
        # put on the correct device
        gt_bbs = [[torch.tensor(ar, dtype=torch.float32).to(device) for ar in fm] for fm in gt_bbs]
        gt_cats = [[torch.tensor(ar, dtype=torch.uint8).to(device) for ar in fm] for fm in gt_cats]
        return ims, gt_bbs, gt_cats

    def data(self):
        if not self._data:
            self._data = json.load((self.filepath/self.pascal_json).open())
        return self._data
    _data = None

    def raw_categories(self, data):
        return {c[ID]:c[NAME] for c in self.data()[CATEGORIES]}

    def categories(self):
        """
        Returns a 0 dict of category id,name including a 'bg' category at the end

        category_ids are 0 indexed. bg category_id is 20
        """
        cats = [x[NAME] for x in self.data()[CATEGORIES]]
        cats.append('bg')
        return {i:c for i,c in enumerate(cats)}

    def category_ids(self):
        return np.array(list(self.categories().keys()))

    def category_names(self):
        return np.array(list(self.categories().values()))

    def raw_annotations(self)->list:
        return self.data()[ANNOTATIONS]

    def raw_images(self):
        return self.data()[IMAGES]

    def images(self):
        # returns a dict of id,image_fullpath
        return {k: f'{config.IMAGE_PATH}/{v}' for k,v in self.get_filenames().items()}

    def get_filenames(self):
        return {o[ID]:o[FILE_NAME] for o in self.raw_images()}

    def get_image_ids(self):
        return list(self.get_filenames())

    def get_annotations(self):
        """
        Returns:
            list<dict> with {
                'image_path': full path to the image,
                'bbs': list of pascal bbs,
                'cats': list of 0 indexed cats
            }
        """
        if not self._annotations:
            raw_ann = self.raw_annotations()
            all_ann = {image_id:{
                IMAGE_PATH: None,
                BBS: [],
                CATS: [],
            } for image_id in self.get_filenames().keys()}

            image_paths = self.images()

            for x in raw_ann:
                image_id = x[IMAGE_ID]
                all_ann[image_id][BBS].append([o for o in x[BBOX]])
                # categories are 0 indexed here
                all_ann[image_id][CATS].append(x[CATEGORY_ID]-1)
                all_ann[image_id][IMAGE_PATH] = image_paths[image_id]

            self._annotations = all_ann
        return self._annotations
    _annotations = None

    def preview(self, data):
        if isinstance(data, (list, tuple)):
            return data[0]
        elif isinstance(data, dict):
            return next(iter(data.items()))
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

    def scaled_im_by_size_and_chw_format(self, im):
        """
        Returns an resized `im` with shape (3, SIZE, SIZE)

        Args:
            im (2d list): HWC format, with it's raw size
        """
        resized_image = cv2.resize(im, (SIZE, SIZE)) # HW
        return np.transpose(resized_image, (2, 0, 1)) # CHW


class TrainPascalDataset(PascalDataset):

    @property
    def pascal_json(self):
        return 'pascal_train2007.json'


class ValPascalDataset(PascalDataset):

    @property
    def pascal_json(self):
        return 'pascal_val2007.json'


class Bboxer:

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
    def hw2corners(cls, center, hw):
        "Return anchor corners based upon center and hw"
        return np.concatenate([center-hw/2, center+hw/2], axis=1)

    @staticmethod
    def aspect_ratios(grid_size):
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
            bbs[:,3]+bbs[:,1]-(1/SIZE),
            bbs[:,2]+bbs[:,0]-(1/SIZE)]).T

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
        for grid_size in FEATURE_MAPS:
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
        bbs = np.multiply(cls.scaled_fastai_bbs(bbs, im), SIZE)
        cats = np.array(cats)
        gt_bbs = bbs[gt_idx]
        gt_cats = cats[gt_idx]
        # the previous line will set the 0 ith class as the gt, so
        # set it to bg if it doesn't meet the IoU threshold
        pos = gt_overlap > THRESH
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
    def scaled_pascal_bbs(bbs, im):
        "Returns scaled pascal bbs scaled [0,1]"
        pascal_bbs = np.array(bbs)
        im_w = im.shape[1]
        im_h = im.shape[0]
        return np.divide(pascal_bbs, [im_w, im_h, im_w, im_h])

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