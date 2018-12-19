import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ssdmultibox.bboxer import Bboxer
from ssdmultibox.config import cfg
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
        self.filepath = cfg.DATA_DIR
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

        gt_bbs, gt_cats = self.bboxer.get_gt_bbs_and_cats_for_all(
            bbs, cats, im)

        return image_id, chw_im, gt_bbs, gt_cats

    @staticmethod
    def to_device(ims, gt_bbs, gt_cats):
        """
        Utility function to put the dataset outputs on the correct device
        """
        ims = torch.tensor(ims, dtype=torch.float32).to(device)
        gt_bbs = torch.tensor(gt_bbs, dtype=torch.float32).to(device)
        gt_cats = torch.tensor(gt_cats, dtype=torch.long).to(device)
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
        return {k: f'{cfg.IMAGE_PATH}/{v}' for k,v in self.get_filenames().items()}

    def get_filenames(self):
        return {o[ID]:o[FILE_NAME] for o in self.raw_images()}

    def get_image_ids(self):
        return list(self.get_filenames())

    def get_image_id_idx_map(self):
        return {x:i for i,x in enumerate(self.get_image_ids())}

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
        resized_image = cv2.resize(im, (SIZE, SIZE)) # HWC
        return np.transpose(resized_image, (2, 0, 1)) # CHW


class TrainPascalDataset(PascalDataset):

    @property
    def pascal_json(self):
        return 'pascal_train2007.json'


class ValPascalDataset(PascalDataset):

    @property
    def pascal_json(self):
        return 'pascal_val2007.json'


class TrainPascalFlatDataset(TrainPascalDataset):
    """
    Dataset where the gt_bbs and gt_cats are a single unique record
    and not duplicated per feature_map/aspect_ratio
    """
    def __getitem__(self, idx):
        """
        Returns:
            image_id (int)
            chw_im (3d array): CHW of image
            gt_bbs (2d array): (n bbs, 4)
            gt_cats (1d array): (n cats,)
        """
        image_ids = self.get_image_ids()
        image_id = image_ids[idx]
        ann = self.get_annotations()[image_id]
        bbs = np.array(ann[BBS])
        cats = np.array(ann[CATS])

        image_paths = self.images()
        im = open_image(image_paths[image_id])
        chw_im = self.scaled_im_by_size_and_chw_format(im)

        gt_bbs, gt_cats = Bboxer.get_stacked_gt(bbs, cats, im)
        gt_bbs *= cfg.NORMALIZED_SIZE

        return image_id, chw_im, gt_bbs, gt_cats


class TransformsTrainPascalFlatDataset(TrainPascalFlatDataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

    def __getitem__(self, idx):
        """
        Returns:
            image_id (int)
            chw_im (3d array): CHW of image
            gt_bbs (2d array): (n bbs, 4)
            gt_cats (1d array): (n cats,)
        """
        image_ids = self.get_image_ids()
        image_id = image_ids[idx]
        ann = self.get_annotations()[image_id]
        bbs = np.array(ann[BBS])
        cats = np.array(ann[CATS])
        image_paths = self.images()
        im = open_image(image_paths[image_id])

        if self.transform:
            im, bbs, cats = self.transform_data(im, bbs, cats)

        chw_im = self.scaled_im_by_size_and_chw_format(im)
        gt_bbs, gt_cats = Bboxer.get_stacked_gt(bbs, cats, im)

        return image_id, chw_im, gt_bbs, gt_cats

    def transform_data(self, im, bbs, cats):
        annotations = {
            'image': im,
            'bboxes': bbs,
            'category_id': cats
        }
        ret = self.transform(**annotations)
        return ret['image'], np.array(ret['bboxes']), np.array(ret['category_id'])
