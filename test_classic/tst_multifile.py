from MFDataset import Dataset
import netCDF4_classic, numpy
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
            if nfile == 0:
                f = netCDF4_classic.Dataset(file,'w',format='NETCDF3_CLASSIC')
            else:
                f = netCDF4_classic.Dataset(file,'w')
            f.createDimension('x',None)
            f.createDimension('y',ydim)
            f.createDimension('z',zdim)
            f.history = 'created today'
            x = f.createVariable('x','i',('x',))
            x.units = 'zlotnys'
            dat = f.createVariable('data','i',('x','y','z',))
            dat.name = 'phony data' 
            nx1 = nfile*ninc; nx2 = ninc*(nfile+1)
            x[0:ninc] = numpy.arange(nfile*ninc,ninc*(nfile+1))
            dat[0:ninc] = data[nx1:nx2]
            f.close()

    def tearDown(self):
        # Remove the temporary files
        for file in self.files:
            os.remove(file)

    def runTest(self):
        """testing multi-file dataset access"""
        f = Dataset(self.files)
        assert f.history == 'created today'
        assert_array_equal(numpy.arange(0,nx),f.variables['x'][:])
        datin = f.variables['data'][4:-4:4,3:5,2:8]
        assert f.variables['data'].name == 'phony data'
        assert_array_equal(datin,data[4:-4:4,3:5,2:8])
        f.close()

if __name__ == '__main__':
    unittest.main()
