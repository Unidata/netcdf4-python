import unittest
import netCDF4

class NCRCTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        """testing access of data over http using opendap"""
        netCDF4.rc_set('foo','bar')
        assert netCDF4.rc_get('foo') == 'bar'

if __name__ == '__main__':
    unittest.main()
