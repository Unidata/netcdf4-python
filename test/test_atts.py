import math
import subprocess
import sys
import unittest
import os
import tempfile
import warnings
import pathlib
import struct
from typing import Literal

import numpy as np
from collections import OrderedDict
from numpy.random.mtrand import uniform

import netCDF4

# test attribute creation.
#FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
FILE_NAME = 'tst_atts.nc'
VAR_NAME="dummy_var"
GROUP_NAME = "dummy_group"
DIM1_NAME="x"
DIM1_LEN=2
DIM2_NAME="y"
DIM2_LEN=3
DIM3_NAME="z"
DIM3_LEN=25
STRATT = 'string attribute'
EMPTYSTRATT = ''
INTATT = 1
FLOATATT = math.pi
SEQATT = np.arange(10)
STRINGSEQATT = ['mary ','','had ','a ','little ','lamb',]
#ATTDICT = {'stratt':STRATT,'floatatt':FLOATATT,'seqatt':SEQATT,
#           'stringseqatt':''.join(STRINGSEQATT), # changed in issue #770
#           'emptystratt':EMPTYSTRATT,'intatt':INTATT}
ATTDICT = {'stratt':STRATT,'floatatt':FLOATATT,'seqatt':SEQATT,
           'stringseqatt':STRINGSEQATT,
           'emptystratt':EMPTYSTRATT,'intatt':INTATT}

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        with netCDF4.Dataset(self.file,'w') as f:
            # try to set a dataset attribute with one of the reserved names.
            f.setncattr('file_format','netcdf4_format')
            # test attribute renaming
            f.stratt_tmp = STRATT
            f.renameAttribute('stratt_tmp','stratt')
            f.emptystratt = EMPTYSTRATT
            f.intatt = INTATT
            f.floatatt = FLOATATT
            f.seqatt = SEQATT
            # sequences of strings converted to a single string.
            f.stringseqatt = STRINGSEQATT
            f.setncattr_string('stringseqatt_array',STRINGSEQATT) # array of NC_STRING
            g = f.createGroup(GROUP_NAME)
            f.createDimension(DIM1_NAME, DIM1_LEN)
            f.createDimension(DIM2_NAME, DIM2_LEN)
            f.createDimension(DIM3_NAME, DIM3_LEN)
            g.createDimension(DIM1_NAME, DIM1_LEN)
            g.createDimension(DIM2_NAME, DIM2_LEN)
            g.createDimension(DIM3_NAME, DIM3_LEN)
            g.stratt_tmp = STRATT
            g.renameAttribute('stratt_tmp','stratt')
            g.emptystratt = EMPTYSTRATT
            g.intatt = INTATT
            g.floatatt = FLOATATT
            g.seqatt = SEQATT
            g.stringseqatt = STRINGSEQATT
            if netCDF4.__version__ > "1.4.2":
                with self.assertRaises(ValueError):
                    g.arrayatt = [[1, 2], [3, 4]] # issue #841
            g.setncattr_string('stringseqatt_array',STRINGSEQATT) # array of NC_STRING
            v = f.createVariable(VAR_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
            # try to set a variable attribute with one of the reserved names.
            v.setncattr('ndim','three')
            v.setncatts({'foo': 1})
            v.setncatts(OrderedDict(bar=2))
            v.stratt_tmp = STRATT
            v.renameAttribute('stratt_tmp','stratt')
            v.emptystratt = EMPTYSTRATT
            v.intatt = INTATT
            v.floatatt = FLOATATT
            v.seqatt = SEQATT
            v.stringseqatt = STRINGSEQATT
            v.setncattr_string('stringseqatt_array',STRINGSEQATT) # array of NC_STRING
            v1 = g.createVariable(VAR_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
            v1.stratt = STRATT
            v1.emptystratt = EMPTYSTRATT
            v1.intatt = INTATT
            v1.floatatt = FLOATATT
            v1.seqatt = SEQATT
            v1.stringseqatt = STRINGSEQATT
            v1.setncattr_string('stringseqatt_array',STRINGSEQATT) # array of NC_STRING
            # issue #959: should not be able to set _FillValue after var creation
            try:
                v1._FillValue(-999.)
            except AttributeError:
                pass
            else:
                raise ValueError('This test should have failed.')
            try:
                v1.setncattr('_FillValue',-999.)
            except AttributeError:
                pass
            else:
                raise ValueError('This test should have failed.')
            # issue #485 (triggers segfault in C lib
            # with version 1.2.1 without pull request #486)
            f.foo = np.array('bar','S')
            f.foo = np.array('bar','U')
            # issue #529 write string attribute as NC_CHAR unless
            # it can't be decoded to ascii.  Add setncattr_string
            # method to force NC_STRING.
            f.charatt = 'foo' # will be written as NC_CHAR
            f.setncattr_string('stringatt','bar') # NC_STRING
            f.cafe = 'caf\xe9' # NC_STRING
            f.batt = 'caf\xe9'.encode() #NC_CHAR
            v.setncattr_string('stringatt','bar') # NC_STRING
            # issue #882 - provide an option to always string attribute
            # as NC_STRINGs. Testing various approaches to setting text attributes...
            f.set_ncstring_attrs(True)
            f.stringatt_ncstr = 'foo' # will now be written as NC_STRING
            f.setncattr_string('stringatt_ncstr','bar') # NC_STRING anyway
            f.caf_ncstr = 'caf\xe9' # NC_STRING anyway
            f.bat_ncstr = 'caf\xe9'.encode() # now NC_STRING
            g.stratt_ncstr = STRATT # now NC_STRING
            #g.renameAttribute('stratt_tmp','stratt_ncstr')
            v.setncattr_string('stringatt_ncstr','bar') # NC_STRING anyway
            v.stratt_ncstr = STRATT
            v1.emptystratt_ncstr = EMPTYSTRATT

    def tearDown(self):
        # Remove the temporary files
        #pass
        os.remove(self.file)

    def runTest(self):
        """testing attributes"""
        with netCDF4.Dataset(self.file, 'r') as f:
            v = f.variables[VAR_NAME]
            g = f.groups[GROUP_NAME]
            v1 = g.variables[VAR_NAME]
            # check attributes in root group.
            # global attributes.
            # check __dict__ method for accessing all netCDF attributes.
            for key,val in ATTDICT.items():
                if type(val) == np.ndarray:
                    assert f.__dict__[key].tolist() == val.tolist()
                else:
                    assert f.__dict__[key] == val
            # check accessing individual attributes.
            assert f.intatt == INTATT
            assert f.floatatt == FLOATATT
            assert f.stratt == STRATT
            assert f.emptystratt == EMPTYSTRATT
            assert f.seqatt.tolist() == SEQATT.tolist()
            #assert f.stringseqatt == ''.join(STRINGSEQATT) # issue 770
            assert f.stringseqatt == STRINGSEQATT
            assert f.stringseqatt_array == STRINGSEQATT
            assert f.getncattr('file_format') == 'netcdf4_format'
            # variable attributes.
            # check __dict__ method for accessing all netCDF attributes.
            for key,val in ATTDICT.items():
                if type(val) == np.ndarray:
                    assert v.__dict__[key].tolist() == val.tolist()
                else:
                    assert v.__dict__[key] == val
            # check accessing individual attributes.
            assert v.intatt == INTATT
            assert v.floatatt == FLOATATT
            assert v.stratt == STRATT
            assert v.seqatt.tolist() == SEQATT.tolist()
            #assert v.stringseqatt == ''.join(STRINGSEQATT) # issue 770
            assert v.stringseqatt == STRINGSEQATT
            assert v.stringseqatt_array == STRINGSEQATT
            assert v.getncattr('ndim') == 'three'
            assert v.getncattr('foo') == 1
            assert v.getncattr('bar') == 2
            # check type of attributes using ncdump (issue #529)
            if not os.getenv('NO_CDL'):
                ncdump_output = f.tocdl()
                for line in ncdump_output:
                    line = line.strip('\t\n\r')
                    line = line.strip()# Must be done another time for group variables
                    if "stringatt" in line: assert line.startswith('string')
                    if "charatt" in line: assert line.startswith(':')
                    if "cafe" in line: assert line.startswith('string')
                    if "batt" in line: assert line.startswith(':')
                    if "_ncstr" in line: assert line.startswith('string')
            # check attributes in subgroup.
            # global attributes.
            for key,val in ATTDICT.items():
                if type(val) == np.ndarray:
                    assert g.__dict__[key].tolist() == val.tolist()
                else:
                    assert g.__dict__[key] == val
            assert g.intatt == INTATT
            assert g.floatatt == FLOATATT
            assert g.stratt == STRATT
            assert g.emptystratt == EMPTYSTRATT
            assert g.seqatt.tolist() == SEQATT.tolist()
            #assert g.stringseqatt == ''.join(STRINGSEQATT) # issue 770
            assert g.stringseqatt == STRINGSEQATT
            assert g.stringseqatt_array == STRINGSEQATT
            for key,val in ATTDICT.items():
                if type(val) == np.ndarray:
                    assert v1.__dict__[key].tolist() == val.tolist()
                else:
                    assert v1.__dict__[key] == val
            assert v1.intatt == INTATT
            assert v1.floatatt == FLOATATT
            assert v1.stratt == STRATT
            assert v1.emptystratt == EMPTYSTRATT
            assert v1.seqatt.tolist() == SEQATT.tolist()
            #assert v1.stringseqatt == ''.join(STRINGSEQATT) # issue 770
            assert v1.stringseqatt == STRINGSEQATT
            assert v1.stringseqatt_array == STRINGSEQATT
            assert getattr(v1,'nonexistantatt',None) == None

        # issue 915 empty string attribute (ncdump reports 'NIL')
        with netCDF4.Dataset(pathlib.Path(__file__).parent / "test_gold.nc") as f:
            assert f['RADIANCE'].VAR_NOTES == ""

class NumericEndianTestCase(unittest.TestCase):

    formats: 'tuple[netCDF4.Format, ...]' = (
        'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF3_64BIT_OFFSET')
    if netCDF4.__has_cdf5_format__ and struct.calcsize('P') >= 8:
        formats += ('NETCDF3_64BIT_DATA',)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = os.path.join(directory.name, 'attributes.nc')

    def test_numeric_attribute_byte_order(self):
        # Issue #1502: nc_put_att expects native-endian numeric buffers.
        for format in self.formats:
            extended = format in ('NETCDF4', 'NETCDF3_64BIT_DATA')
            dtypes = ['i1', 'i2', 'i4', 'i8', 'f4', 'f8']
            if extended:
                dtypes += ['u1', 'u2', 'u4', 'u8']
            expected = {}
            with netCDF4.Dataset(self.path, 'w', format=format) as f:
                targets: dict[str, netCDF4.Dataset | netCDF4.Group | netCDF4.Variable] = {
                    '/': f, '/v': f.createVariable('v', 'f8')}
                if format == 'NETCDF4':
                    group = f.createGroup('g')
                    targets['/g'] = group
                    targets['/g/v'] = group.createVariable('v', 'f8')
                for dtype in dtypes:
                    values = [1, 2, 17] if dtype[0] == 'u' else [1, -2, 17]
                    if dtype[0] == 'f':
                        values = [1.25, -2.5, 17.75]
                    for order in ('=', '<', '>'):
                        array = np.array(values, dtype=np.dtype(dtype).newbyteorder(order))
                        readonly = array.copy()
                        readonly.flags.writeable = False
                        inputs = (array, np.array(values[0], dtype=array.dtype),
                                  np.repeat(array, 2)[::2], array[::-1],
                                  readonly, array[:0])
                        for layout, value in enumerate(inputs):
                            original = value.tobytes()
                            for target_name, target in targets.items():
                                for setter in ('single', 'bulk', 'syntax'):
                                    name = '%s_%s_%s_%s' % (dtype, order, layout, setter)
                                    with self.subTest(format=format, target=target_name,
                                                      attribute=name):
                                        if setter == 'single':
                                            target.setncattr(name, value)
                                        elif setter == 'bulk':
                                            target.setncatts({name: value})
                                        else:
                                            setattr(target, name, value)
                                        self.assertEqual(value.tobytes(), original)
                                        self.assertEqual(value.dtype, array.dtype)
                                        self.assertEqual(value.flags.writeable,
                                                         layout != 4)
                                    stored_dtype = 'i4' if dtype == 'i8' and not extended else dtype
                                    expected[target_name, name] = value.astype(stored_dtype)
            with netCDF4.Dataset(self.path) as f:
                for (target_name, name), value in expected.items():
                    target = f if target_name == '/' else f[target_name]
                    with self.subTest(format=format, target=target_name, attribute=name):
                        actual = np.asarray(target.getncattr(name))
                        np.testing.assert_array_equal(actual, value)
                        self.assertEqual(actual.dtype, value.dtype)

    def test_valid_range_byte_order(self):
        for format in self.formats:
            with self.subTest(format=format):
                value = np.array([1., 2.], dtype=np.dtype('f8').newbyteorder('S'))
                with netCDF4.Dataset(self.path, 'w', format=format) as f:
                    f.createDimension('x', 4)
                    v = f.createVariable('v', 'f8', ('x',))
                    v[:] = [0., 1., 2., 3.]
                    v.setncattr('valid_range', value)
                    f.setncattr('levels', value)
                with netCDF4.Dataset(self.path) as f:
                    v = f['v']
                    np.testing.assert_array_equal(f.levels, [1., 2.])
                    np.testing.assert_array_equal(v.valid_range, [1., 2.])
                    np.testing.assert_array_equal(np.ma.getmaskarray(v[:]),
                                                  [True, False, False, True])
                    np.testing.assert_array_equal(v[:].compressed(), [1., 2.])

    def test_variable_metadata_byte_order(self):
        # Existing fill-value and special-attribute callers must not swap twice.
        endians: tuple[Literal['little', 'big'], ...] = ('little', 'big')
        for endian in endians:
            with self.subTest(endian=endian):
                dtype = np.dtype('f8').newbyteorder(endian)
                with netCDF4.Dataset(self.path, 'w') as f:
                    f.createDimension('x', 3)
                    v = f.createVariable('v', dtype, ('x',), endian=endian,
                                         fill_value=-999.)
                    v[:2] = [1., 2.]
                    v.valid_min = np.array(0., dtype=dtype)
                    v.valid_max = np.array(3., dtype=dtype)
                    v.valid_range = np.array([0., 3.], dtype=dtype)
                    v.missing_value = np.array(-999., dtype=dtype)
                with netCDF4.Dataset(self.path) as f:
                    v = f['v']
                    self.assertEqual(v._FillValue, -999.)
                    self.assertEqual(v.missing_value, -999.)
                    self.assertEqual(v.valid_min, 0.)
                    self.assertEqual(v.valid_max, 3.)
                    np.testing.assert_array_equal(v.valid_range, [0., 3.])
                    np.testing.assert_array_equal(np.ma.getmaskarray(v[:]),
                                                  [False, False, True])
                    np.testing.assert_array_equal(v[:].compressed(), [1., 2.])

    def test_unsupported_numeric_attributes(self):
        with netCDF4.Dataset(self.path, 'w') as f:
            for dtype in ('?', '<f2', '>f2', '<c8', '>c8', '<c16', '>c16'):
                with self.subTest(dtype=dtype):
                    with self.assertRaises(TypeError):
                        f.setncattr('unsupported', np.array([1, 0], dtype=dtype))


if __name__ == '__main__':
    unittest.main()
