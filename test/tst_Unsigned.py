import unittest
import netCDF4
from numpy.testing import assert_array_equal
import numpy as np

class Test_Unsigned(unittest.TestCase):
    """
    Test autoconversion to unsigned ints when _Unsigned attribute is True.
    This attribute is is set by netcdf-java to designate unsigned
    integer data stored with a signed integer type in netcdf-3.
    If _Unsigned=True, a view to the data as unsigned integers is returned.
    set_autoscale can be used to turn this off (default is on)
    See issue #656 (pull reqeust #658).
    """
    def test_unsigned(self):
        f = netCDF4.Dataset("ubyte.nc")
        data = f['ub'][:]
        assert data.dtype.str[1:] == 'u1'
        assert_array_equal(data,np.array([0,255],np.uint8))
        f.set_auto_scale(False)
        data2 = f['ub'][:]
        assert data2.dtype.str[1:] == 'i1'
        assert_array_equal(data2,np.array([0,-1],np.int8))
        f.close()

if __name__ == '__main__':
    unittest.main()
