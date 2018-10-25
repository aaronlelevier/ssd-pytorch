import json

import cv2
import numpy as np
import torch
from fastai.dataset import open_image
from torch.utils.data import Dataset

from ssdmultibox import config

SIZE = 224

NUM_CLASSES = 21

# IoU threshold
THRESH = 0.5

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

    def __init__(self, grid_size, k=1):
        self.filepath = config.DATADIR
        self.bboxer = Bboxer(grid_size, k)

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

        gt_bbs, gt_cats = self.bboxer.get_gt_bbs_and_cats(bbs, cats, im)

        return image_id, chw_im, gt_bbs, gt_cats

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

    def scaled_bbs_by_size(self, bbs):
        """
        Returns an resized `bbs` by the SIZE

        Args:
            bbs (1d list): normalized size of [0,1]
        """
        return np.multiply(bbs, SIZE)


class TrainPascalDataset(PascalDataset):

    @property
    def pascal_json(self):
        return 'pascal_train2007.json'


class Bboxer:

    def __init__(self, grid_size, k=1):
        self.grid_size = grid_size

    def anchors(self):
        anc_grid = 4
        k = 1
        anc_offset = 1/(anc_grid*2)
        anc_x = np.repeat(np.linspace(anc_offset, 1-anc_offset, anc_grid), 4)
        anc_y = np.tile(np.linspace(anc_offset, 1-anc_offset, anc_grid), 4)
        anc_centers = np.tile(np.stack([anc_x,anc_y], axis=1), (k,1))
        anc_w = 1/anc_grid
        anc_h = 1/anc_grid
        anc_sizes = np.array([[anc_w, anc_h] for i in range(anc_grid*anc_grid)])
        return np.array(
            np.concatenate([anc_centers, anc_sizes], axis=1), dtype=np.float)

    def anchor_corners(self):
        anchors = self.anchors()
        return self.hw2corners(anchors[:,:2], anchors[:,2:])

    def hw2corners(self, center, hw):
        return np.concatenate([center-hw/2, center+hw/2], axis=1)

    def get_intersection(self, bbs, im):
        # returns the i part of IoU scaled [0,1]
        bbs = self.scaled_fastai_bbs(bbs, im)
        bbs16 = np.reshape(np.tile(bbs, 16), (-1,16,4))
        anchor_corners = self.anchor_corners()
        intersect = np.abs(np.maximum(
            anchor_corners[:,:2], bbs16[:,:,:2]) - np.minimum(anchor_corners[:,2:], bbs16[:,:,2:]))
        return intersect[:,:,0] * intersect[:,:,1]

    def scaled_fastai_bbs(self, bbs, im):
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

    def get_ancb_area(self):
        "Returns the [0,1] normalized area of a single anchor box"
        return 1. / np.square(self.grid_size)

    def get_bbs_area(self, bbs, im):
        "Returns a np.array of the [0,1] normalized bbs area"
        bbs = self.scaled_fastai_bbs(bbs, im)
        return np.abs(bbs[:,0]-bbs[:,2])*np.abs(bbs[:,1]-bbs[:,3])

    def get_iou(self, bbs, im):
        intersect = self.get_intersection(bbs, im)
        bbs_union = self.get_ancb_area() + self.get_bbs_area(bbs, im)
        return (intersect.T / bbs_union).T

    def get_gt_overlap_and_idx(self, bbs, im):
        overlaps = torch.tensor(self.get_iou(bbs, im))
        _, prior_idx = overlaps.max(1)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        # sets the gt_idx equal to the obj_idx for each prior_idx
        for i,o in enumerate(prior_idx):
            gt_idx[o] = i
        return gt_overlap, gt_idx

    def get_gt_bbs_and_cats(self, bbs, cats, im):
        """
        Returns bbs per anchor box gt labels and
        1 hot encoded category per anchor box

        Args:
            bbs (2d list): of pascal bbs int coordinates
            cats (2d list):
                sparse list of int categories, one-hot encoded with the final
                'bg' category shaved off
                NOTE: nuance here, this allows the model to predict that there
                are no objects identified, but it's behavior isn't guided
                towards predicting the 'bg' everytime just to naively minimize
                the cost
            im: (2d list): of HWC raw pascal im
        Returns:
            bbs (2d list), cats (1d list)
        """
        gt_overlap, gt_idx = self.get_gt_overlap_and_idx(bbs, im)
        bbs = np.multiply(self.scaled_fastai_bbs(bbs, im), SIZE)
        cats = np.array(cats)
        gt_bbs = bbs[gt_idx]
        gt_cats = cats[gt_idx]
        # the previous line will set the 0 ith class as the gt, so
        # set it to bg if it doesn't meet the IoU threshold
        pos = gt_overlap > THRESH
        neg_idx = np.nonzero(1-pos)[:,0]
        gt_cats[neg_idx] = 20
        gt_cats = self.one_hot_encode(gt_cats, NUM_CLASSES)[:,:-1]
        return np.reshape(gt_bbs, (-1)), np.reshape(gt_cats, (-1))

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

    @staticmethod
    def pascal_bbs(bbs):
        """
        Returns fastai_bbs as pascal formatted bbs

        Args:
            bbs (2d list): fastai encoded bbs
        """
        return np.array([bbs[:,1],bbs[:,0],bbs[:,3]-bbs[:,1]+1,bbs[:,2]-bbs[:,0]+1]).T
