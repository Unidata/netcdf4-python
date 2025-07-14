import unittest
import netCDF4
from netCDF4 import __has_nc_rc_set__

class NCRCTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        """test rc_get, rc_set functions"""
        if __has_nc_rc_set__:
            netCDF4.rc_set('foo','bar')
            assert netCDF4.rc_get('foo') == 'bar'
            assert netCDF4.rc_get('bar') == None

if __name__ == '__main__':
    unittest.main()
