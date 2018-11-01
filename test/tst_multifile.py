from netCDF4 import Dataset, MFDataset, MFTime, num2date, date2num, date2index
import numpy as np
from numpy.random import seed, randint
from numpy.testing import assert_array_equal, assert_equal
from numpy import ma
import tempfile, unittest, os, datetime
import cftime
from pkg_resources import parse_version

nx=100; ydim=5; zdim=10
nfiles = 10
ninc = nx/nfiles
files = [tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name for nfile in range(nfiles)]
data = randint(0,10,size=(nx,ydim,zdim))
missval = 99
data[::10] = missval
data = ma.masked_values(data,missval)

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
            dat.long_name = 'phony data'
            dat.missing_value = missval
            nx1 = int(nfile*ninc); nx2 = int(ninc*(nfile+1))
            #x[0:ninc] = np.arange(nfile*ninc,ninc*(nfile+1))
            x[:] = np.arange(nfile*ninc,ninc*(nfile+1))
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
        f.set_auto_maskandscale(True) # issue570
        f.set_always_mask(False)
        assert f.history == 'created today'
        assert_array_equal(np.arange(0,nx),f.variables['x'][:])
        varin = f.variables['data']
        datin = varin[:]
        assert_array_equal(datin.mask,data.mask)
        varin.set_auto_maskandscale(False)
        data2 = data.filled()
        assert varin.long_name == 'phony data'
        assert len(varin) == nx
        assert varin.shape == (nx,ydim,zdim)
        assert varin.dimensions == ('x','y','z')
        assert_array_equal(varin[4:-4:4,3:5,2:8],data2[4:-4:4,3:5,2:8])
        assert varin[0,0,0] == data2[0,0,0]
        assert_array_equal(varin[:],data2)
        assert getattr(varin,'nonexistantatt',None) == None
        f.close()
        # test master_file kwarg (issue #835).
        f = MFDataset(self.files,master_file=self.files[-1],check=True)
        assert_array_equal(np.arange(0,nx),f.variables['x'][:])
        varin = f.variables['data']
        assert_array_equal(varin[4:-4:4,3:5,2:8],data2[4:-4:4,3:5,2:8])
        f.close()
        # testing multi-file get_variables_by_attributes
        f = MFDataset(self.files,check=True)
        assert f.get_variables_by_attributes(axis='T') == []
        f.get_variables_by_attributes(units='zlotys')[0] == f['x']
        f.close()

class NonuniformTimeTestCase(unittest.TestCase):
    ninc = 365
    def setUp(self):

        self.files = [tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name for nfile in range(2)]
        for nfile,file in enumerate(self.files):
            f = Dataset(file,'w',format='NETCDF4_CLASSIC')
            f.createDimension('time',None)
            f.createDimension('y',ydim)
            f.createDimension('z',zdim)
            f.history = 'created today'

            time = f.createVariable('time', 'f', ('time', ))
            #time.units = 'days since {0}-01-01'.format(1979+nfile)
            yr = 1979+nfile
            time.units = 'days since %s-01-01' % yr

            # Do not set the calendar attribute on the created files to test calendar
            # overload.
            # time.calendar = 'standard'

            x = f.createVariable('x','f',('time', 'y', 'z'))
            x.units = 'potatoes per square mile'

            nx1 = self.ninc*nfile;
            nx2 = self.ninc*(nfile+1)

            time[:] = np.arange(self.ninc)
            x[:] = np.arange(nx1, nx2).reshape(self.ninc,1,1) * np.ones((1, ydim, zdim))

            f.close()

    def tearDown(self):
        # Remove the temporary files
        for file in self.files:
            os.remove(file)


    def runTest(self):
        # The test files have no calendar attribute on the time variable.
        calendar = 'standard'

        # Get the real dates
        # skip this until cftime pull request #55 is in a released
        # version (1.0.1?). Otherwise, fix for issue #808 breaks this
        if parse_version(cftime.__version__) >= parse_version('1.0.1'):
            dates = []
            for file in self.files:
                f = Dataset(file)
                t = f.variables['time']
                dates.extend(num2date(t[:], t.units, calendar))
                f.close()

        # Compare with the MF dates
        f = MFDataset(self.files,check=True)
        t = f.variables['time']

        T = MFTime(t, calendar=calendar)
        assert_equal(T.calendar, calendar)
        assert_equal(len(T), len(t))
        assert_equal(T.shape, t.shape)
        assert_equal(T.dimensions, t.dimensions)
        assert_equal(T.typecode(), t.typecode())
        # skip this until cftime pull request #55 is in a released
        # version (1.0.1?). Otherwise, fix for issue #808 breaks this
        if parse_version(cftime.__version__) >= parse_version('1.0.1'):
            assert_array_equal(num2date(T[:], T.units, T.calendar), dates)
        assert_equal(date2index(datetime.datetime(1980, 1, 2), T), 366)
        f.close()

        # Test exception is raised when no calendar attribute is available on the
        # time variable.
        with MFDataset(self.files, check=True) as ds:
            with self.assertRaises(ValueError):
                MFTime(ds.variables['time'])

        # Test exception is raised when the calendar attribute is different on the
        # variables. First, add calendar attributes to file. Note this will modify
        # the files inplace.
        calendars = ['standard', 'gregorian']
        for idx, f in enumerate(self.files):
            with Dataset(f, 'a') as ds:
                ds.variables['time'].calendar = calendars[idx]
        with MFDataset(self.files, check=True) as ds:
            with self.assertRaises(ValueError):
                MFTime(ds.variables['time'])


if __name__ == '__main__':
    unittest.main()
