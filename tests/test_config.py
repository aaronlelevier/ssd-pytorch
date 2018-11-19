import os
import unittest

from ssdmultibox import config


class ConfigTests(unittest.TestCase):

    def test_data_dir(self):
        home_dir = os.path.expanduser('~')

        ret = config.DATA_DIR

        if home_dir.startswith('/Users/alelevier'):
            assert str(ret) == f'{home_dir}/data'
        elif home_dir.startswith('/Users/aaron'):
            assert str(ret) == f'{home_dir}/data/VOC2007/trainval/VOCdevkit/VOC2007'
