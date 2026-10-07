"""Empty orthogonal selections retain all unselected dimensions."""
import os
import tempfile
import unittest
from typing import Any

import numpy as np
from numpy.testing import assert_array_equal
from netCDF4 import Dataset, MFDataset


EMPTY = np.array([], dtype=np.intp)
SHAPE = (3, 5, 7)
CASES = [
    (EMPTY, slice(None), slice(None)),
    (slice(None), EMPTY, slice(None)),
    (slice(None), slice(None), EMPTY),
    (np.zeros(3, dtype=bool), slice(None), slice(None)),
    (slice(None), np.zeros(5, dtype=bool), slice(None)),
    (slice(None), slice(None), np.zeros(7, dtype=bool)),
    (EMPTY, slice(None, None, -1), slice(None, None, 2)),
    (slice(None, None, 2), EMPTY, slice(None, None, -2)),
    (1, EMPTY, slice(None, None, -1)),
    (EMPTY, 2, slice(None)),
    (slice(None), EMPTY, 3),
    (EMPTY, EMPTY, slice(None)),
    (EMPTY, slice(None), EMPTY),
    (EMPTY, EMPTY, EMPTY),
    (EMPTY, [4, 1, 4], [6, 2]),
    ([2, 0, 2], EMPTY, [6, 1, 6]),
    (slice(0), slice(None), slice(None)),
    (slice(None), slice(0), slice(None)),
    (slice(None), slice(None), slice(0)),
    ([2, 0, 2], [4, 1, 4], [6, 2]),
    (slice(None), slice(None), slice(None)),
    (1, slice(None, None, -1), slice(None, None, 2)),
]


def orthogonal_selection(data, key):
    # Apply one axis at a time, so NumPy cannot broadcast multiple index arrays.
    result = data
    for axis in range(len(key) - 1, -1, -1):
        index = key[axis]
        if isinstance(index, slice):
            single = [slice(None)] * result.ndim
            single[axis] = index
            result = result[tuple(single)]
        else:
            index = np.asarray(index)
            if index.dtype.kind == 'b':
                index = np.flatnonzero(index)
            result = np.take(result, index, axis=axis)
    return result


class EmptyIndexingTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, 'empty.nc')
        self.sources: dict[str, np.ndarray] = {}
        with Dataset(self.path, 'w') as ds:
            for name, length in zip(('x', 'y', 'z'), SHAPE):
                ds.createDimension(name, length)
            for dtype in ('i4', 'f8'):
                source = np.arange(np.prod(SHAPE), dtype=dtype).reshape(SHAPE)
                source[1, 2, 3] = -9999
                self.sources[dtype] = np.ma.masked_equal(source, -9999)
                variable = ds.createVariable(dtype, dtype, ('x', 'y', 'z'), fill_value=-9999)
                variable[:] = source
            strings = np.array(['s%d' % n for n in range(np.prod(SHAPE))], dtype=object).reshape(SHAPE)
            ds.createVariable('strings', str, ('x', 'y', 'z'))[:] = strings
            self.sources['strings'] = strings
            vlen_type = ds.createVLType(np.int32, 'integers')
            vlens = np.empty(SHAPE, dtype=object)
            for index in np.ndindex(SHAPE):
                vlens[index] = np.arange(sum(index) + 1, dtype=np.int32)
            ds.createVariable('vlens', vlen_type, ('x', 'y', 'z'))[:] = vlens
            self.sources['vlens'] = vlens

    def tearDown(self):
        self.directory.cleanup()

    def assert_selection(self, variable, source, key):
        expected = orthogonal_selection(source, key)
        actual = variable[key]
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        if actual.dtype == object:
            for index in np.ndindex(expected.shape):
                assert_array_equal(actual[index], expected[index])
        else:
            assert_array_equal(actual, expected)
            assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))

    def test_reads(self):
        # Reopen to exercise reads from the actual C extension and disk.
        with Dataset(self.path) as ds:
            for name, source in self.sources.items():
                for use_get_vars in (True, False):
                    variable = ds[name]
                    variable.use_nc_get_vars(use_get_vars)
                    for key in CASES:
                        with self.subTest(variable=name, use_get_vars=use_get_vars, key=repr(key)):
                            self.assert_selection(variable, source, key)
            self.assertEqual(tuple(len(ds.dimensions[n]) for n in ('x', 'y', 'z')), SHAPE)

    def test_empty_writes(self):
        # Both a correctly shaped empty RHS and a scalar are no-ops.
        with Dataset(self.path, 'r+') as ds:
            for name in ('i4', 'f8'):
                variable = ds[name]
                for key in CASES:
                    expected = orthogonal_selection(self.sources[name], key)
                    if expected.size:
                        continue
                    with self.subTest(variable=name, key=repr(key)):
                        variable[key] = expected
                        variable[key] = 123
                assert_array_equal(variable[:], self.sources[name])
                self.assertEqual(variable.shape, SHAPE)
        with Dataset(self.path) as ds:
            for name in ('i4', 'f8'):
                assert_array_equal(ds[name][:], self.sources[name])
                self.assertEqual(ds[name].shape, SHAPE)

    def test_empty_unlimited_dimension(self):
        path = os.path.join(self.directory.name, 'unlimited.nc')
        with Dataset(path, 'w') as ds:
            ds.createDimension('time', None)
            ds.createDimension('range', 7)
            variable = ds.createVariable('data', 'i4', ('time', 'range'))
            for key in ((EMPTY, slice(None)), (np.zeros(0, dtype=bool), slice(None))):
                with self.subTest(key=repr(key)):
                    self.assertEqual(variable[key].shape, (0, 7))
                    variable[key] = np.empty((0, 7), dtype='i4')
                    variable[key] = 123
                    self.assertEqual(variable.shape, (0, 7))
            variable[:] = np.arange(21, dtype='i4').reshape(3, 7)
            variable[EMPTY, :] = 123
            self.assertEqual(variable.shape, (3, 7))
            assert_array_equal(variable[:], np.arange(21, dtype='i4').reshape(3, 7))

    def test_multifile_reads(self):
        source = np.arange(6 * 5 * 7, dtype='i4').reshape(6, 5, 7)
        paths = []
        for number in range(2):
            path = os.path.join(self.directory.name, 'part%d.nc' % number)
            paths.append(path)
            with Dataset(path, 'w', format='NETCDF4_CLASSIC') as ds:
                ds.createDimension('time', None)
                ds.createDimension('y', 5)
                ds.createDimension('z', 7)
                variable = ds.createVariable('data', 'i4', ('time', 'y', 'z'))
                variable[:] = source[number * 3:(number + 1) * 3]
        with MFDataset(paths) as ds:
            for key in (
                (EMPTY, slice(None), slice(None)),
                (np.zeros(6, dtype=bool), slice(None), slice(None)),
                (slice(None), EMPTY, slice(None)),
                (slice(None), slice(None), EMPTY),
                (EMPTY, EMPTY, slice(None)),
                (EMPTY, 2, slice(None, None, 2)),
                (2, EMPTY, slice(None)),
                (slice(0), slice(None), slice(None)),
                (slice(None), slice(None), slice(None)),
                ([5, 1, 5], slice(None), [6, 2]),
            ):
                with self.subTest(key=repr(key)):
                    self.assert_selection(ds['data'], source, key)

    def test_empty_writes_read_only(self):
        # Existing empty-array writes also work on read-only variables.
        with Dataset(self.path) as ds:
            for name in ('i4', 'f8'):
                for key in CASES[:16]:
                    with self.subTest(variable=name, key=repr(key)):
                        ds[name][key] = 123
                assert_array_equal(ds[name][:], self.sources[name])

    def test_empty_fancy_indices_use_bounded_metadata(self):
        from netCDF4.utils import _StartCountStride

        with Dataset(self.path) as ds:
            for length in (10, 20):
                repeated = np.zeros(length, dtype=np.intp)
                key = (EMPTY, repeated, repeated)
                with self.subTest(length=length):
                    metadata = _StartCountStride(key, SHAPE)
                    # Empty output must not allocate the Cartesian product of
                    # repeated indices. This bound permits linear metadata.
                    self.assertLessEqual(sum(a.nbytes for a in metadata), 128 * length)
                    self.assertEqual(ds['i4'][key].shape, (0, length, length))
            with self.assertRaises(IndexError):
                ds['i4'][EMPTY, [99], [0]]

    def test_empty_broadcast_writes(self):
        with Dataset(self.path, 'r+') as ds:
            for name in ('i4', 'f8'):
                variable = ds[name]
                for empty in (EMPTY, slice(0), np.zeros(3, dtype=bool)):
                    for rhs in (np.array([1, 2]), np.ones((3, 2))):
                        with self.subTest(variable=name, empty=repr(empty), shape=rhs.shape):
                            variable[empty, [4, 1, 4], [6, 2]] = rhs
                assert_array_equal(variable[:], self.sources[name])
                self.assertEqual(variable.shape, SHAPE)

    def test_multifile_empty_decoding(self):
        paths = []
        raw = np.arange(12, dtype='i2').reshape(4, 3)
        attributes: dict[str, dict[str, Any]] = {
            'scaled': {'scale_factor': np.float64(0.5)},
            'offset': {'add_offset': np.float64(2.0)},
            'identity': {'scale_factor': np.float32(1), 'add_offset': np.float32(0)},
            'unsigned': {'_Unsigned': 'true'},
        }
        for number in range(2):
            path = os.path.join(self.directory.name, 'packed%d.nc' % number)
            paths.append(path)
            with Dataset(path, 'w', format='NETCDF4_CLASSIC') as ds:
                ds.createDimension('time', None)
                ds.createDimension('y', 3)
                for name, attrs in attributes.items():
                    variable = ds.createVariable(name, 'i2', ('time', 'y'))
                    variable.setncatts(attrs)
                    variable.set_auto_maskandscale(False)
                    variable[:] = raw[number * 2:(number + 1) * 2]
        with MFDataset(paths) as ds:
            for name, attrs in attributes.items():
                for scale in (True, False):
                    for mask in (True, False):
                        variable = ds[name]
                        variable.set_auto_scale(scale)
                        variable.set_auto_mask(mask)
                        expected = raw
                        if scale:
                            if name == 'unsigned':
                                expected = raw.astype('u2')
                            elif name == 'identity':
                                expected = raw.astype('f4')
                            else:
                                expected = raw * attrs.get('scale_factor', 1) + attrs.get('add_offset', 0)
                        for key in ((EMPTY, slice(None)), (slice(0), slice(None)),
                                    (slice(None), EMPTY), (slice(None), slice(0)),
                                    (1, EMPTY), (slice(None), slice(None))):
                            with self.subTest(variable=name, scale=scale, mask=mask, key=repr(key)):
                                self.assert_selection(variable, expected, key)

    def test_empty_scalar_bounds(self):
        with Dataset(self.path) as ds:
            for name in self.sources:
                for empty in (EMPTY, slice(0), np.zeros(3, dtype=bool)):
                    for index in (99, [99], -99):
                        with self.subTest(variable=name, empty=repr(empty), index=repr(index)):
                            with self.assertRaises(IndexError):
                                ds[name][empty, index, :]

    def test_one_dimensional_empty_broadcast(self):
        path = os.path.join(self.directory.name, 'vector.nc')
        with Dataset(path, 'w') as ds:
            ds.createDimension('x', 3)
            variable = ds.createVariable('data', 'i4', ('x',))
            variable[:] = [10, 20, 30]
            for empty in (EMPTY, slice(0), np.zeros(3, dtype=bool)):
                with self.subTest(empty=repr(empty)):
                    variable[empty] = np.array([123])
                    assert_array_equal(variable[:], [10, 20, 30])
                    with self.assertRaises((ValueError, IndexError)):
                        variable[empty] = np.array([123, 456])

    def test_empty_encoded_characters(self):
        raw = np.array(['alpha', 'bravo', 'delta', 'gamma'], dtype='S5').view('S1').reshape(4, 5)
        paths = []
        for number, values in enumerate((raw, raw[:2], raw[2:])):
            path = os.path.join(self.directory.name, 'chars%d.nc' % number)
            paths.append(path)
            with Dataset(path, 'w', format='NETCDF4_CLASSIC') as ds:
                ds.createDimension('time', None)
                ds.createDimension('width', 5)
                variable = ds.createVariable('data', 'S1', ('time', 'width'))
                variable._Encoding = 'utf-8'
                variable.set_auto_chartostring(False)
                variable[:] = values
        keys = [
            ((EMPTY, slice(None)), True),
            ((slice(0), slice(None)), True),
            ((EMPTY, slice(1, 4)), False),
            ((EMPTY, slice(0)), False),
            ((slice(None), EMPTY), False),
            ((slice(None), slice(0)), False),
            ((1, EMPTY), False),
            ((EMPTY, 2), False),
            ((EMPTY, [4, 1, 4, 0, 2]), False),
            ((slice(0), [0, 2, 1, 4, 3]), False),
            ((slice(None), slice(1, 4)), False),
            ((slice(None), [4, 1, 4, 0, 2]), False),
        ]
        with Dataset(paths[0]) as regular, MFDataset(paths[1:]) as multi:
            for route, ds in (('regular', regular), ('multifile', multi)):
                variable = ds['data']
                for convert in (True, False):
                    variable.set_auto_chartostring(convert)
                    for key, full_width in keys:
                        with self.subTest(route=route, convert=convert, key=repr(key)):
                            expected = orthogonal_selection(raw, key)
                            if convert and full_width:
                                # All full-width cases here have empty records.
                                expected = np.empty(expected.shape[:-1], dtype='U5')
                            actual = variable[key]
                            self.assertEqual(actual.shape, expected.shape)
                            self.assertEqual(actual.dtype, expected.dtype)
                            assert_array_equal(actual, expected)

    def test_empty_scalar_character_width_one(self):
        paths = []
        for number in range(2):
            path = os.path.join(self.directory.name, 'onechar%d.nc' % number)
            paths.append(path)
            with Dataset(path, 'w', format='NETCDF4_CLASSIC') as ds:
                ds.createDimension('time', None)
                ds.createDimension('width', 1)
                variable = ds.createVariable('data', 'S1', ('time', 'width'))
                variable._Encoding = 'utf-8'
                variable.set_auto_chartostring(False)
                variable[:] = np.array([['a'], ['b']], dtype='S1')
        with Dataset(paths[0]) as regular, MFDataset(paths) as multi:
            for route, ds in (('regular', regular), ('multifile', multi)):
                for empty in (EMPTY, slice(0)):
                    with self.subTest(route=route, empty=repr(empty)):
                        actual = ds['data'][empty, 0]
                        self.assertEqual(actual.shape, (0,))
                        self.assertEqual(actual.dtype, np.dtype('S1'))

    def test_multifile_selected_decoding(self):
        paths = []
        raw = np.arange(18, dtype='i2').reshape(6, 3)
        attributes: dict[str, tuple[dict[str, Any], ...]] = {
            'first_float': ({'scale_factor': np.float64(0.5)}, {}, {}),
            'last_float': ({}, {}, {'scale_factor': np.float32(0.5)}),
            'unsigned': ({'_Unsigned': 'true'}, {}, {'_Unsigned': 'true'}),
        }
        for number in range(3):
            path = os.path.join(self.directory.name, 'different%d.nc' % number)
            paths.append(path)
            with Dataset(path, 'w', format='NETCDF4_CLASSIC') as ds:
                ds.createDimension('time', None)
                ds.createDimension('y', 3)
                for name, attrs in attributes.items():
                    variable = ds.createVariable(name, 'i2', ('time', 'y'))
                    variable.setncatts(attrs[number])
                    variable.set_auto_maskandscale(False)
                    variable[:] = raw[number * 2:(number + 1) * 2]

        # Slices concatenate contributing files and promote their dtypes.
        # Existing non-slice fancy reads cast the result for each record chunk,
        # so their final dtype is the dtype of the last selected record.
        # With no selected records, retain the first-file decoding convention.
        selections = (
            (2, (1,)), (4, (2,)), (-1, (2,)),
            (slice(None), (0, 1, 2)),
            (slice(2, 6), (1, 2)), (slice(1, 5), (0, 1, 2)),
            (slice(None, None, 3), (0, 1)),
            ([0, 4, 2], (1,)), ([4, 0, 4], (2,)),
            ([0, 0, 0], (0,)), ([4, 4, 4], (2,)),
            (np.array([True, False, True, False, True, False]), (0, 1, 2)),
            (EMPTY, (0,)), (slice(0), (0,)),
        )
        # check=True validates structure but permits per-file attributes.
        with MFDataset(paths, check=True) as ds:
            for name, attrs in attributes.items():
                variable = ds[name]
                for scale in (True, False):
                    variable.set_auto_scale(scale)
                    parts = []
                    for number in range(3):
                        part = raw[number * 2:(number + 1) * 2]
                        if scale:
                            if '_Unsigned' in attrs[number]:
                                part = part.astype('u2')
                            elif 'scale_factor' in attrs[number]:
                                part = part * attrs[number]['scale_factor']
                        parts.append(part)
                    for mask in (True, False):
                        variable.set_auto_mask(mask)
                        for records, contributors in selections:
                            expected_dtype = np.result_type(*(parts[n].dtype for n in contributors))
                            for empty in (EMPTY, slice(0)):
                                with self.subTest(variable=name, scale=scale, mask=mask,
                                                  records=repr(records), empty=repr(empty)):
                                    expected_shape = orthogonal_selection(raw, (records, empty)).shape
                                    for key in ((records, empty), (Ellipsis, records, empty)):
                                        actual = variable[key]
                                        self.assertEqual(actual.shape, expected_shape)
                                        self.assertEqual(actual.dtype, expected_dtype)
                                        self.assertEqual(actual.size, 0)
                                    if isinstance(records, slice) and records == slice(None):
                                        actual = variable[..., empty]
                                        self.assertEqual(actual.shape, expected_shape)
                                        self.assertEqual(actual.dtype, expected_dtype)
                            if isinstance(records, (int, np.integer)):
                                with self.subTest(control=name, scale=scale, mask=mask, records=records):
                                    record = int(records) % 6
                                    actual = variable[records, :]
                                    expected = parts[record // 2][record % 2]
                                    self.assertEqual(actual.dtype, expected.dtype)
                                    assert_array_equal(actual, expected)

    def test_multifile_selected_character_decoding(self):
        paths = []
        raw = np.full((6, 2, 3), b'a', dtype='S1')
        for number, encoding in enumerate(('utf-8', None, 'bytes')):
            path = os.path.join(self.directory.name, 'encodings%d.nc' % number)
            paths.append(path)
            with Dataset(path, 'w', format='NETCDF4_CLASSIC') as ds:
                ds.createDimension('time', None)
                ds.createDimension('y', 2)
                ds.createDimension('width', 3)
                variable = ds.createVariable('data', 'S1', ('time', 'y', 'width'))
                if encoding is not None:
                    variable._Encoding = encoding
                variable.set_auto_chartostring(False)
                variable[:] = raw[number * 2:(number + 1) * 2]
        cases = (
            (0, 'U3'), (2, None), (4, 'S3'),
            (slice(2, 4), None), (slice(None, None, 4), 'U3'),
            ([2, 0, 2], None), ([4, 0, 4], 'S3'), (EMPTY, 'U3'),
        )
        with MFDataset(paths, check=True) as ds:
            variable = ds['data']
            for convert in (True, False):
                variable.set_auto_chartostring(convert)
                for records, decoded_dtype in cases:
                    for empty in (EMPTY, slice(0)):
                        with self.subTest(records=repr(records), convert=convert, empty=repr(empty)):
                            expected = orthogonal_selection(raw, (records, empty, slice(None)))
                            if convert and decoded_dtype is not None:
                                expected = np.empty(expected.shape[:-1], dtype=decoded_dtype)
                            actual = variable[records, empty, :]
                            self.assertEqual(actual.shape, expected.shape)
                            self.assertEqual(actual.dtype, expected.dtype)
                            self.assertEqual(actual.size, 0)
                if convert:
                    # Some contributors decode the character axis and others
                    # do not. Existing concatenation rejects mixed ranks.
                    with self.assertRaises(ValueError):
                        variable[:, :0, :]
                for record in (0, 2, 4):
                    assert_array_equal(variable[record, :, 0], raw[record, :, 0])


if __name__ == '__main__':
    unittest.main()
