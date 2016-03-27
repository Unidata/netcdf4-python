import sys
import unittest
import os
import tempfile
import numpy as NP
import netCDF4

# test implicit group creation by using unix-like paths
# in createVariable and createGroups (added in 1.1.8).
# also test Dataset.__getitem__, also added in 1.1.8.

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

class Groups2TestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = netCDF4.Dataset(self.file,'w')
        x = f.createDimension('x',10)
        # create groups in path if they don't already exist
        v = f.createVariable('/grouped/data/v',float,('x',))
        g = f.groups['grouped']
        # create groups underneath 'grouped'
        v2 = g.createVariable('./data/data2/v2',float,('x',))
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing implicit group and creation and Dataset.__getitem__"""
        f  = netCDF4.Dataset(self.file, 'r')
        v1 = f['/grouped/data/v']
        v2 = ((f.groups['grouped']).groups['data']).variables['v']
        g = f['/grouped/data']
        v3 = g['data2/v2']
        assert(v1 == v2)
        assert(g == f.groups['grouped'].groups['data'])
        assert(v3.name == 'v2')
        f.close()

if __name__ == '__main__':
    unittest.main()
