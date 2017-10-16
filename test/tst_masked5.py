import unittest
import os
import tempfile

import numpy as np
from numpy import ma
from numpy.testing import assert_array_equal
from netCDF4 import Dataset

# Test use of vector of missing values.

class VectorMissingValues(unittest.TestCase):

    def setUp(self):

        self.testfile = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
        self.testfile2 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

        self.missing_values = [-999,999,0]
        self.v    = np.array([-999,0,1,2,3,999], dtype = "i2")
        self.v_ma = ma.array([-1,0,1,2,3,4], dtype = "i2", \
                    mask = [True, True, False, False, False, True])

        f = Dataset(self.testfile, 'w')
        d = f.createDimension('x',6)
        v = f.createVariable('v', "i2", 'x')

        v.missing_value = self.missing_values
        v[:] = self.v

        f.close()
        f2 = Dataset(self.testfile2, 'w',format='NETCDF3_CLASSIC')
        d2 = f.createDimension('x',6)
        v2 = f.createVariable('v', "i4", 'x')
        f2.close()


    def tearDown(self):

        os.remove(self.testfile)


    def test_scaled(self):

        """Testing auto-conversion of masked arrays"""

        f = Dataset(self.testfile)
        v = f.variables["v"]
        self.assertTrue(isinstance(v[:], ma.core.MaskedArray))
        assert_array_equal(v[:], self.v_ma)
        assert_array_equal(v[2],self.v[2]) # issue #624.
        v.set_auto_mask(False)
        self.assertTrue(isinstance(v[:], np.ndarray))
        assert_array_equal(v[:], self.v)
        f.close()

        # issue 725
        f = Dataset(self.testfile2)
        # all elements should be masked since no data written
        assert(f.variables['v'][:].all() is ma.masked)
        f.close()


if __name__ == '__main__':
    unittest.main()
