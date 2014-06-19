import unittest
import os
import tempfile

import numpy as np
from numpy import ma
from numpy.testing import assert_array_almost_equal
from netCDF4 import Dataset, default_fillvals

# Test automatic conversion of masked arrays (set_auto_mask())

class SetAutoMaskTestBase(unittest.TestCase):

    """Base object for tests checking the functionality of set_auto_mask()"""

    def setUp(self):

        self.testfile = tempfile.mktemp(".nc")

        self.fillval = default_fillvals["i2"]
        self.v    = np.array([self.fillval, 5, 4, -9999], dtype = "i2")
        self.v_ma = ma.array([self.fillval, 5, 4, -9999], dtype = "i2", mask = [True, False, False, True])

        self.scale_factor = 10.
        self.add_offset = 5.

        self.v_scaled = self.v * self.scale_factor + self.add_offset
        self.v_ma_scaled = self.v_ma * self.scale_factor + self.add_offset

        f = Dataset(self.testfile, 'w')
        _ = f.createDimension('x', None)
        v = f.createVariable('v', "i2", 'x')

        v.missing_value = np.array(-9999, v.dtype)

        # v[0] not set, will be equal to _FillValue
        v[1] = self.v[1]
        v[2] = self.v[2]
        v[3] = v.missing_value

        f.close()


    def tearDown(self):

        os.remove(self.testfile)


class SetAutoMaskFalse(SetAutoMaskTestBase):

    def test_unscaled(self):

        """Testing auto-conversion of masked arrays for set_auto_mask(False)"""

        f = Dataset(self.testfile, "r")

        f.variables["v"].set_auto_mask(False)
        v = f.variables["v"][:]

        self.assertEqual(v.dtype, "i2")
        self.assertIsInstance(v, np.ndarray)
        self.assertNotIsInstance(v, ma.core.MaskedArray)
        assert_array_almost_equal(v, self.v)

        f.close()


    def test_scaled(self):

        """Testing auto-conversion of masked arrays for set_auto_mask(False) with scaling"""

        # Update test data file

        f = Dataset(self.testfile, "a")
        f.variables["v"].scale_factor = self.scale_factor
        f.variables["v"].add_offset = self.add_offset
        f.close()

        # Note: Scaling variables is default if scale_factor and/or add_offset are present

        f = Dataset(self.testfile, "r")

        f.variables["v"].set_auto_mask(False)
        v = f.variables["v"][:]

        self.assertEqual(v.dtype, "f8")
        self.assertIsInstance(v, np.ndarray)
        self.assertNotIsInstance(v, ma.core.MaskedArray)
        assert_array_almost_equal(v, self.v_scaled)

        f.close()


class SetAutoMaskTrue(SetAutoMaskTestBase):

    def test_unscaled(self):

        """Testing auto-conversion of masked arrays for set_auto_mask(True)"""

        f = Dataset(self.testfile)

        f.variables["v"].set_auto_mask(True) # The default anyway...
        v_ma = f.variables['v'][:]

        self.assertEqual(v_ma.dtype, "i2")
        self.assertIsInstance(v_ma, np.ndarray)
        self.assertIsInstance(v_ma, ma.core.MaskedArray)
        assert_array_almost_equal(v_ma, self.v_ma)
        f.close()

    def test_scaled(self):

        """Testing auto-conversion of masked arrays for set_auto_mask(True)"""

        # Update test data file

        f = Dataset(self.testfile, "a")
        f.variables["v"].scale_factor = self.scale_factor
        f.variables["v"].add_offset = self.add_offset
        f.close()

        # Note: Scaling variables is default if scale_factor and/or add_offset are present

        f = Dataset(self.testfile)

        f.variables["v"].set_auto_mask(True)  # The default anyway...
        v_ma = f.variables['v'][:]

        self.assertEqual(v_ma.dtype, "f8")
        self.assertIsInstance(v_ma, np.ndarray)
        self.assertIsInstance(v_ma, ma.core.MaskedArray)
        assert_array_almost_equal(v_ma, self.v_ma_scaled)
        f.close()


if __name__ == '__main__':
    unittest.main()
