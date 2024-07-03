import sys
import unittest
import os
import tempfile
from netCDF4 import Dataset
import numpy as np
from numpy.testing import assert_array_equal

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
VL_NAME = 'vlen_type'
VL_BASETYPE = np.int16
DIM1_NAME = 'lon'
DIM2_NAME = 'lat'
nlons = 5; nlats = 5
VAR1_NAME = 'ragged'
VAR2_NAME = 'strings'
VAR3_NAME = 'strings_alt'
VAR4_NAME = 'string_scalar'
VAR5_NAME = 'vlen_scalar'
data = np.empty(nlats*nlons,object)
datas = np.empty(nlats*nlons,object)
nn = 0
for n in range(nlats*nlons):
    nn = nn + 1
    data[n] = np.arange(nn,dtype=VL_BASETYPE)
    datas[n] = ''.join([chr(i) for i in range(97,97+nn+1)])
data = np.reshape(data,(nlats,nlons))
datas = np.reshape(datas,(nlats,nlons))


class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = Dataset(self.file,'w')
        vlen_t = f.createVLType(VL_BASETYPE, VL_NAME)
        f.createDimension(DIM1_NAME,nlons)
        f.createDimension(DIM2_NAME,nlats)
        ragged = f.createVariable(VAR1_NAME, vlen_t,\
                (DIM2_NAME,DIM1_NAME))
        strings = f.createVariable(VAR2_NAME, str,
                (DIM2_NAME,DIM1_NAME))
        strings_alt = f.createVariable(VAR3_NAME, datas.astype(str).dtype,
                                       (DIM2_NAME, DIM1_NAME))
        string_scalar = f.createVariable(VAR4_NAME,str,())
        vlen_scalar = f.createVariable(VAR5_NAME,vlen_t,())
        ragged[:] = data
        ragged[-1,-1] = data[-1,-1]
        strings[:] = datas
        strings[-2,-2] = datas[-2,-2]
        strings_alt[:] = datas.astype(str)
        string_scalar[...] = 'foo'  #issue458
        vlen_scalar[...] = np.array([1,2,3],np.int16)
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing vlen variables"""
        f = Dataset(self.file, 'r')
        v = f.variables[VAR1_NAME]
        vs = f.variables[VAR2_NAME]
        vs_alt = f.variables[VAR3_NAME]
        assert list(f.vltypes.keys()) == [VL_NAME]
        assert f.vltypes[VL_NAME].dtype == VL_BASETYPE
        assert f.variables['string_scalar'][...] == 'foo'
        assert_array_equal(f.variables['vlen_scalar'][...],np.array([1,2,3],np.int16))
        data2 = v[:]
        data2s = vs[:]
        # issue #1306
        assert repr(vs[[0,2,3],0]) == "array(['ab', 'abcdefghijkl', 'abcdefghijklmnopq'], dtype=object)" 
        for i in range(nlons):
            for j in range(nlats):
                assert_array_equal(data2[j,i], data[j,i])
                assert datas[j,i] == data2s[j,i]
        assert_array_equal(datas, vs_alt[:])
        f.close()


class TestInvalidDataType(unittest.TestCase):
    def runTest(self):
        f = Dataset(FILE_NAME, 'w', format='NETCDF3_CLASSIC')
        f.createDimension('x', 1)
        # using assertRaisesRegext as a context manager
        # only works with python >= 2.7 (issue #497)
        #with self.assertRaisesRegexp(ValueError, 'strings are only supported'):
        #    f.createVariable('foo', str, ('x',))
        try:
            f.createVariable('foo', str, ('x',))
        except ValueError:
            pass
        f.close()
        os.remove(FILE_NAME)

class TestScalarVlenString(unittest.TestCase):
    # issue 333
    def runTest(self):
        f = Dataset(FILE_NAME, 'w', format='NETCDF4')
        teststring = f.createVariable('teststring', str)
        stringout = "yyyymmdd_hhmmss"
        teststring[()] = stringout
        f.close()
        f = Dataset(FILE_NAME)
        assert f.variables['teststring'][:] == stringout
        f.close()
        os.remove(FILE_NAME)

class TestIntegerIndex(unittest.TestCase):
    # issue 526
    def runTest(self):
        strtest = Dataset(FILE_NAME, 'w', format='NETCDF4')
        strtest.createDimension('tenstrings', 10)
        strtest.createVariable('tenstrings', str, ['tenstrings'])
        strtest['tenstrings'][np.int32(5)] = 'asdf'
        strtest['tenstrings'][6.0] = 'asdf'
        strtest.close()
        f = Dataset(FILE_NAME)
        assert f.variables['tenstrings'][np.int32(5)] == 'asdf'
        assert f.variables['tenstrings'][6.0] == 'asdf'
        f.close()
        os.remove(FILE_NAME)

class TestObjectArrayIndexing(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = Dataset(self.file,'w')
        vlen_t = f.createVLType(VL_BASETYPE, VL_NAME)
        f.createDimension(DIM1_NAME,nlons)
        f.createDimension(DIM2_NAME,nlats)
        strings_alt = f.createVariable(VAR3_NAME, datas.astype(str).dtype,
                                       (DIM2_NAME, DIM1_NAME))
        strings_alt[:] = datas.astype(str)
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing vlen variables"""
        f = Dataset(self.file, 'r')
        vs_alt = f.variables[VAR3_NAME]
        unicode_strings = vs_alt[:]
        fancy_indexed = unicode_strings[0][[1,2,4]]
        assert fancy_indexed[0] == 'abc'
        assert fancy_indexed[1] == 'abcd'
        assert fancy_indexed[2] == 'abcdef'
        f.close()

class VlenAppendTestCase(unittest.TestCase):
    def setUp(self):

        import netCDF4
        if netCDF4.__netcdf4libversion__ < "4.4.1":
            self.skip = True
            try:
                self.skipTest("This test requires NetCDF 4.4.1 or later.")
            except AttributeError:
                # workaround for Python 2.6 (skipTest(reason) is new
                # in Python 2.7)
                pass
        else:
            self.skip = False

        self.file = FILE_NAME
        f = Dataset(self.file, 'w')
        vlen_type = f.createVLType(np.float64, 'vltest')
        f.createDimension('x', None)
        v = f.createVariable('vl', vlen_type, 'x')
        w = f.createVariable('vl2', np.float64, 'x')
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing appending to vlen variables (issue #527)."""
        # workaround for Python 2.6
        if self.skip:
            return

        f = Dataset(self.file, 'a')
        w = f.variables["vl2"]
        v = f.variables["vl"]
        w[0:3] = np.arange(3, dtype=np.float64)
        v[0]                    # sometimes crashes
        v[0].tolist()           # sometimes crashes
        v[0].size               # BOOM!
        f.close()

class Vlen_ScaledInts(unittest.TestCase):
    def setUp(self):
        self.file = FILE_NAME
        nc = Dataset(self.file, 'w')
        vlen_type = nc.createVLType(np.uint8, 'vltest')
        nc.createDimension('x', None)
        v = nc.createVariable('vl', vlen_type, 'x')
        v.scale_factor = 1./254.
        v.missing_value=np.array(255,np.uint8)
        # random lengths between 1 and 1000
        ilen = np.random.randint(1,1000,size=100)
        n = 0
        for nlen in ilen:
            data = np.random.uniform(low=0.0, high=1.0, size=nlen)
            v[n] = data
            if n==99: self.data = data
            n += 1
        nc.close()
    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)
    def runTest(self):
        """testing packing float vlens as scaled integers (issue #1003)."""
        nc = Dataset(self.file)
        data = nc['vl'][-1]
        # check max error of compression
        err = np.abs(data - self.data)
        assert err.max() < nc['vl'].scale_factor 
        # turn off auto-scaling
        nc.set_auto_maskandscale(False)
        data = nc['vl'][-1]
        assert data[-1] == np.around(self.data[-1]/nc['vl'].scale_factor) 
        nc.close()

if __name__ == '__main__':
    unittest.main()
