from torch.utils.data import DataLoader

from ssdmultibox.datasets import TrainPascalDataset
from ssdmultibox.models import SSDModel


class ModelAndDatasetSetupMixin:

    @classmethod
    def setUpClass(cls):
        # data
        dataset = TrainPascalDataset()
        dataloader = DataLoader(dataset, batch_size=4, num_workers=0)
        image_ids, ims, gt_bbs, gt_cats = next(iter(dataloader))
        ims, cls.gt_bbs, cls.gt_cats = dataset.to_device(ims, gt_bbs, gt_cats)
        # model
        model = SSDModel()
        cls.preds = model(ims)

    def setUp(self):
        self.gt_bbs = self.__class__.gt_bbs
        self.gt_cats = self.__class__.gt_cats
        self.preds = self.__class__.preds
