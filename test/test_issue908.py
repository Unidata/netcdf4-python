import netCDF4, unittest
import numpy as np
import pathlib


class Issue908TestCase(unittest.TestCase):

    def setUp(self):
        self.nc = netCDF4.Dataset(pathlib.Path(__file__).parent / "CRM032_test1.nc")

    def tearDown(self):
        self.nc.close()

    def runTest(self):
        data = self.nc['rgrid'][:]
        assert data.all() is np.ma.masked 

if __name__ == '__main__':
    unittest.main()
