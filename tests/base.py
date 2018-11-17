import unittest

import numpy as np


class BaseTestCase(unittest.TestCase):

    def assert_arr_equals(self, ret, raw_ret):
        assert np.isclose(
                np.array(ret, dtype=np.float16),
                np.array(raw_ret, dtype=np.float16)
            ).all(), f"\nret:\n{ret}\nraw_ret:\n{raw_ret}"
