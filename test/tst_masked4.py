import unittest
import os
import tempfile

import numpy as np
from numpy import ma
from numpy.testing import assert_array_almost_equal
from netCDF4 import Dataset, default_fillvals

# Test use of valid_min/valid_max/valid_range in generation of masked arrays

class SetValidMinMax(unittest.TestCase):

    def setUp(self):

        self.testfile = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

        self.valid_min = -32765
        self.valid_max = 32765
        self.valid_range = [self.valid_min,self.valid_max]
        self.v    = np.array([self.valid_min-1, 5, 4, self.valid_max+1], dtype = "i2")
        self.v_ma = ma.array([self.valid_min-1, 5, 4, self.valid_max+1], dtype = "i2", mask = [True, False, False, True])

        self.scale_factor = 10.
        self.add_offset = 5.

        self.v_scaled = self.v * self.scale_factor + self.add_offset
        self.v_ma_scaled = self.v_ma * self.scale_factor + self.add_offset

        f = Dataset(self.testfile, 'w')
        _ = f.createDimension('x', None)
        v = f.createVariable('v', "i2", 'x')
        v2 = f.createVariable('v2', "i2", 'x')
        v3 = f.createVariable('v3', "i2", 'x', fill_value=self.valid_min)

        v.missing_value = np.array(32767, v.dtype)
        v.valid_min = np.array(self.valid_min, v.dtype)
        v.valid_max = np.array(self.valid_max, v.dtype)

        v[0] = self.valid_min-1
        v[1] = self.v[1]
        v[2] = self.v[2]
        v[3] = self.valid_max+1

        v2.missing_value = np.array(32767, v.dtype)
        v2.valid_range = np.array(self.valid_range, v.dtype)

        v2[0] = self.valid_range[0]-1
        v2[1] = self.v[1]
        v2[2] = self.v[2]
        v2[3] = self.valid_range[1]+1

        v3.missing_value = np.array(32767, v.dtype)
        v3.valid_max = np.array(self.valid_max, v.dtype)

        # _FillValue should act as valid_min
        v3[0] = v3._FillValue-1
        v3[1] = self.v[1]
        v3[2] = self.v[2]
        v3[3] = self.valid_max+1

        f.close()


    def tearDown(self):

        os.remove(self.testfile)


    def test_scaled(self):

        """Testing auto-conversion of masked arrays"""

        # Update test data file

        f = Dataset(self.testfile, "a")
        f.variables["v"].scale_factor = self.scale_factor
        f.variables["v"].add_offset = self.add_offset
        f.variables["v2"].scale_factor = self.scale_factor
        f.variables["v2"].add_offset = self.add_offset
        f.close()

        f = Dataset(self.testfile, "r")
        v = f.variables["v"][:]
        v2 = f.variables["v2"][:]
        v3 = f.variables["v3"][:]
        self.assertEqual(v.dtype, "f8")
        self.assertTrue(isinstance(v, np.ndarray))
        self.assertTrue(isinstance(v, ma.core.MaskedArray))
        assert_array_almost_equal(v, self.v_scaled)
        self.assertEqual(v2.dtype, "f8")
        self.assertTrue(isinstance(v2, np.ndarray))
        self.assertTrue(isinstance(v2, ma.core.MaskedArray))
        assert_array_almost_equal(v2, self.v_scaled)
        self.assertTrue(np.all(self.v_ma.mask == v.mask))
        self.assertTrue(np.all(self.v_ma.mask == v2.mask))
        # treating _FillValue as valid_min/valid_max was
        # too suprising, revert to old behaviour (issue #761)
        #self.assertTrue(np.all(self.v_ma.mask == v3.mask))
        # check that underlying data is same as in netcdf file
        v = f.variables['v']
        v.set_auto_scale(False) 
        v = v[:]
        self.assertTrue(np.all(self.v == v.data))
        f.close()

        # issue 672
        f = Dataset('issue672.nc')
        field = 'azi_angle_trip'
        v = f.variables[field]
        data1 = v[:]
        v.set_auto_scale(False)
        data2 = v[:]
        v.set_auto_maskandscale(False)
        data3 = v[:]
        assert(data1[(data3 < v.valid_min)].mask.sum() == 12)
        assert(data2[(data3 < v.valid_min)].mask.sum() ==
               data1[(data3 < v.valid_min)].mask.sum())
        f.close()


if __name__ == '__main__':
    unittest.main()
