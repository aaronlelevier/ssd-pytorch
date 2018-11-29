from ssdmultibox.datasets import NUM_CLASSES
from tests.base import ModelAndDatasetBaseTestCase


class SSDModelTests(ModelAndDatasetBaseTestCase):

    def test_model_output_sizes(self):
        assert isinstance(self.preds, tuple)
        assert len(self.preds) == 2
        assert self.preds[0].shape == (4, 11640, 4)
        assert self.preds[1].shape == (4, 11640, NUM_CLASSES)
