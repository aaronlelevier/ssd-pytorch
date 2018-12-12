from ssdmultibox.datasets import NUM_CLASSES
from tests.base import ModelAndDatasetBaseTestCase


class SSDModelTests(ModelAndDatasetBaseTestCase):

    def test_model_output_sizes(self):
        assert isinstance(self.preds, tuple)
        assert len(self.preds) == 2
        assert self.preds[0].shape == (4, 11640, 4)
        assert self.preds[1].shape == (4, 11640, NUM_CLASSES)

    def test_unfreeze(self):
        # base network starts out frozen
        for i, layer in enumerate(self.model.parameters()):
            if i < 58:
                assert layer.requires_grad == False
            else:
                assert layer.requires_grad == True

        self.model.unfreeze()

        # all layers now trainable
        for i, layer in enumerate(self.model.parameters()):
            assert layer.requires_grad
