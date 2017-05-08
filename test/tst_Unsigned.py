import os
import unittest
import netCDF4
from numpy.testing import assert_array_equal
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class Test_Unsigned(unittest.TestCase):
    def test_unsigned(self):
        fpath = os.path.join(CURRENT_DIR, "ubyte.nc3")
        f = netCDF4.Dataset(fpath)
        data = f['ub'][:]
        assert data.dtype.str[1:] == 'u1'
        assert_array_equal(data,np.array([0,255],np.uint8))
        f.close()

if __name__ == '__main__':
    unittest.main()
