import numpy as np

from ssdmultibox.datasets.v1 import TrainPascalDataset,  Bboxer, SIZE
from ssdmultibox.utils import open_image

BBS = 'bbs'
CATS = 'cats'

class TrainPascalDataset(TrainPascalDataset):

    def __getitem__(self, idx):
        image_ids = self.get_image_ids()
        image_id = image_ids[idx]
        ann = self.get_annotations()[image_id]
        bbs = ann[BBS]
        cats = ann[CATS]

        image_paths = self.images()
        im = open_image(image_paths[image_id])
        chw_im = self.scaled_im_by_size_and_chw_format(im)
        gt_bbs = Bboxer.scaled_fastai_bbs(bbs, im) * SIZE
        gt_cats = np.array(cats)

        return image_id, chw_im, gt_bbs, gt_cats