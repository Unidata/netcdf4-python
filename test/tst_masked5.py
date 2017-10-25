import unittest
import os
import tempfile

import numpy as np
from numpy import ma
from numpy.testing import assert_array_equal
from netCDF4 import Dataset, __netcdf4libversion__

# Test use of vector of missing values.

class VectorMissingValues(unittest.TestCase):

    def setUp(self):

        self.testfile = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

        self.missing_values = [-999,999,0]
        self.v    = np.array([-999,0,1,2,3,999], dtype = "i2")
        self.v_ma = ma.array([-1,0,1,2,3,4], dtype = "i2", \
                    mask = [True, True, False, False, False, True])

        f = Dataset(self.testfile, 'w')
        d = f.createDimension('x',6)
        v = f.createVariable('v', "i2", 'x')
        # issue 730: set fill_value for vlen str vars
        v2 = f.createVariable('v2',str,'x',fill_value=u'<missing>')

        v.missing_value = self.missing_values
        v[:] = self.v
        v2[0]='first'

        f.close()


    def tearDown(self):

        os.remove(self.testfile)


    def test_scaled(self):

        """Testing auto-conversion of masked arrays"""

        f = Dataset(self.testfile)
        v = f.variables["v"]
        v2 = f.variables["v2"]
        self.assertTrue(isinstance(v[:], ma.core.MaskedArray))
        assert_array_equal(v[:], self.v_ma)
        assert_array_equal(v[2],self.v[2]) # issue #624.
        v.set_auto_mask(False)
        self.assertTrue(isinstance(v[:], np.ndarray))
        assert_array_equal(v[:], self.v)

        # issue 730
        # this part fails with netcdf 4.1.3
        # a bug in vlen strings?
        if __netcdf4libversion__ >= '4.4.0':
            assert (v2[0]==u'first')
            assert (v2[1]==u'<missing>')


        f.close()


if __name__ == '__main__':
    unittest.main()
