import unittest
import os
import tempfile

import numpy as np
from numpy import ma
from numpy.testing import assert_array_almost_equal

from netCDF4 import Dataset, default_fillvals

# Test automatic scaling of variables (set_auto_scale())

class SetAutoScaleTestBase(unittest.TestCase):

    """Base object for tests checking the functionality of set_auto_scale()"""

    def setUp(self):

        self.testfile = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

        self.fillval = default_fillvals["i2"]
        self.missing_value = -9999

        self.v    = np.array([0, 5, 4, self.missing_value], dtype = "i2")
        self.v_ma = ma.array([0, 5, 4, self.missing_value], dtype = "i2",
                             mask = [True, False, False, True], fill_value = self.fillval)

        self.scale_factor = 10.
        self.add_offset = 5.

        self.v_scaled = self.v * self.scale_factor + self.add_offset
        self.v_ma_scaled = self.v_ma * self.scale_factor + self.add_offset

        f = Dataset(self.testfile, 'w')
        x = f.createDimension('x', None)
        v = f.createVariable('v', "i2", 'x')

        v[:] = self.v

        # Note: Scale factors are only added after writing, so that no auto-scaling takes place!

        v.scale_factor = self.scale_factor
        v.add_offset = self.add_offset

        f.close()


    def tearDown(self):

        os.remove(self.testfile)


class SetAutoScaleFalse(SetAutoScaleTestBase):

    def test_unmasked(self):

        """Testing (not) auto-scaling of variables for set_auto_scale(False)"""

        f = Dataset(self.testfile, "r")

        f.variables["v"].set_auto_scale(False)
        v = f.variables["v"][:]

        self.assertEqual(v.dtype, "i2")
        self.assertTrue(isinstance(v, np.ndarray))
        # issue 785: always return masked array by default
        self.assertTrue(isinstance(v, ma.core.MaskedArray))
        assert_array_almost_equal(v, self.v)

        f.close()


    def test_masked(self):

        """Testing auto-conversion of masked arrays for set_auto_mask(False) with masking"""

        # Update test data file

        f = Dataset(self.testfile, "a")
        f.variables["v"].missing_value = self.missing_value
        f.close()

        # Note: Converting arrays to masked arrays is default if missing_value is present

        f = Dataset(self.testfile, "r")

        f.variables["v"].set_auto_scale(False)
        v_ma = f.variables["v"][:]

        self.assertEqual(v_ma.dtype, "i2")
        self.assertTrue(isinstance(v_ma, np.ndarray))
        self.assertTrue(isinstance(v_ma, ma.core.MaskedArray))
        assert_array_almost_equal(v_ma, self.v_ma)

        f.close()


class SetAutoScaleTrue(SetAutoScaleTestBase):

    def test_unmasked(self):

        """Testing auto-scaling of variables for set_auto_scale(True)"""

        f = Dataset(self.testfile)

        f.variables["v"].set_auto_scale(True) # The default anyway...
        v_scaled = f.variables['v'][:]

        self.assertEqual(v_scaled.dtype, "f8")
        self.assertTrue(isinstance(v_scaled, np.ndarray))
        # issue 785: always return masked array by default
        self.assertTrue(isinstance(v_scaled, ma.core.MaskedArray))
        assert_array_almost_equal(v_scaled, self.v_scaled)
        f.close()

    def test_masked(self):

        """Testing auto-scaling of variables for set_auto_scale(True) with masking"""

        # Update test data file

        f = Dataset(self.testfile, "a")
        f.variables["v"].missing_value = self.missing_value
        f.close()

        # Note: Converting arrays to masked arrays is default if missing_value is present

        f = Dataset(self.testfile)

        f.variables["v"].set_auto_scale(True)  # The default anyway...
        v_ma_scaled = f.variables['v'][:]

        self.assertEqual(v_ma_scaled.dtype, "f8")
        self.assertTrue(isinstance(v_ma_scaled, np.ndarray))
        self.assertTrue(isinstance(v_ma_scaled, ma.core.MaskedArray))
        assert_array_almost_equal(v_ma_scaled, self.v_ma_scaled)
        f.close()


class WriteAutoScaleTest(SetAutoScaleTestBase):

    def test_auto_scale_write(self):

        """Testing automatic packing to all kinds of integer types"""

        def packparams(dmax, dmin, dtyp):
            kind = dtyp[0]
            n = int(dtyp[1]) * 8
            scale_factor = (dmax - dmin) / (2**n - 1)
            if kind == 'i':
                add_offset = dmin + 2**(n-1) * scale_factor
            elif kind == 'u':
                add_offset = dmin
            else:
                raise Exception
            return((add_offset, scale_factor))

        for dtyp in ['i1', 'i2', 'i4', 'u1', 'u2', 'u4']:
            np.random.seed(456)
            data = np.random.uniform(size=100)
            f = Dataset(self.testfile, 'w')
            f.createDimension('x')
            #
            # save auto_scaled
            v = f.createVariable('v', dtyp, ('x',))
            v.set_auto_scale(True)  # redundant
            v.add_offset, v.scale_factor = packparams(
                np.max(data), np.min(data), dtyp)
            v[:] = data
            f.close()
            #
            # read back
            f = Dataset(self.testfile, 'r')
            v = f.variables['v']
            v.set_auto_mask(False)
            v.set_auto_scale(True)  # redundant
            vdata = v[:]
            # error normalized by scale factor
            maxerrnorm = np.max(np.abs((vdata - data) / v.scale_factor))
            # 1e-5 accounts for floating point errors
            assert(maxerrnorm < 0.5 + 1e-5)
            f.close()


class GlobalSetAutoScaleTest(unittest.TestCase):

    def setUp(self):

        self.testfile = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

        f = Dataset(self.testfile, 'w')

        grp1 = f.createGroup('Group1')
        grp2 = f.createGroup('Group2')
        f.createGroup('Group3')         # empty group

        f.createVariable('var0', "i2", ())
        grp1.createVariable('var1', 'f8', ())
        grp2.createVariable('var2', 'f4', ())

        f.close()

    def tearDown(self):

        os.remove(self.testfile)

    def runTest(self):

        f = Dataset(self.testfile, "r")

        # Default is both scaling and masking enabled

        v0 = f.variables['var0']
        v1 = f.groups['Group1'].variables['var1']
        v2 = f.groups['Group2'].variables['var2']

        self.assertTrue(v0.scale)
        self.assertTrue(v0.mask)

        self.assertTrue(v1.scale)
        self.assertTrue(v1.mask)

        self.assertTrue(v2.scale)
        self.assertTrue(v2.mask)

        # No auto-scaling

        f.set_auto_scale(False)

        self.assertFalse(v0.scale)
        self.assertTrue(v0.mask)

        self.assertFalse(v1.scale)
        self.assertTrue(v1.mask)

        self.assertFalse(v2.scale)
        self.assertTrue(v2.mask)

        f.close()


if __name__ == '__main__':
    unittest.main()
