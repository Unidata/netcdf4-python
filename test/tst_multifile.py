from netCDF4 import Dataset, MFDataset
import numpy
from numpy.random import seed, randint
from numpy.testing import assert_array_equal
import tempfile, unittest, os

nx=100; ydim=5; zdim=10
nfiles = 10
ninc = nx/nfiles
files = [tempfile.mktemp(".nc") for nfile in range(nfiles)]
data = randint(0,10,size=(nx,ydim,zdim))

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.files = files
        for nfile,file in enumerate(self.files):
            f = Dataset(file,'w',format='NETCDF4_CLASSIC')
            f.createDimension('x',None)
            f.createDimension('y',ydim)
            f.createDimension('z',zdim)
            f.history = 'created today'
            x = f.createVariable('x','i',('x',))
            x.units = 'zlotys'
            dat = f.createVariable('data','i',('x','y','z',))
            dat.name = 'phony data' 
            nx1 = nfile*ninc; nx2 = ninc*(nfile+1)
            #x[0:ninc] = numpy.arange(nfile*ninc,ninc*(nfile+1))
            x[:] = numpy.arange(nfile*ninc,ninc*(nfile+1))
            #dat[0:ninc] = data[nx1:nx2]
            dat[:] = data[nx1:nx2]
            f.close()

    def tearDown(self):
        # Remove the temporary files
        for file in self.files:
            os.remove(file)

    def runTest(self):
        """testing multi-file dataset access"""
        f = MFDataset(self.files,check=True)
        assert f.history == 'created today'
        assert_array_equal(numpy.arange(0,nx),f.variables['x'][:])
        varin = f.variables['data']
        assert varin.name == 'phony data'
        assert varin.shape == (nx,ydim,zdim)
        assert varin.dimensions == ('x','y','z')
        assert_array_equal(varin[4:-4:4,3:5,2:8],data[4:-4:4,3:5,2:8])
        assert varin[0,0,0] == data[0,0,0]
        f.close()

if __name__ == '__main__':
    unittest.main()
