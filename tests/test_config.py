import os
import unittest

import pytest

from ssdmultibox import config
from ssdmultibox.config import cfg
from ssdmultibox.criterion import SSDLoss


class ConfigTests(unittest.TestCase):

    def test_data_dir(self):
        home_dir = os.path.expanduser('~')

        ret = cfg.DATA_DIR

        if home_dir.startswith('/Users/alelevier'):
            assert str(ret) == f'{home_dir}/data'
        elif home_dir.startswith('/Users/aaron'):
            assert str(ret) == f'{home_dir}/data/VOC2007/trainval/VOCdevkit/VOC2007'

    def test_config_has_defaults(self):
        assert cfg.SSD_LOSS_ALPHA == 1
        assert cfg.NMS_OVERLAP == 0.5

    def test_config_update_defaults(self):
        new_loss_alpha = 2

        cfg.update({'SSD_LOSS_ALPHA': new_loss_alpha})

        assert cfg.SSD_LOSS_ALPHA == new_loss_alpha

    def test_config_raise_error_if_not_a_valid_config_key(self):
        assert not hasattr(cfg, 'foo')

        with pytest.raises(config.InvalidConfigError):
            cfg.update({'foo': 1})
