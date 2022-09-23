import numpy as np
from netCDF4 import set_alignment, get_alignment, Dataset
from netCDF4 import __has_set_alignment__
import netCDF4
import os
import subprocess
import tempfile
import unittest

# During testing, sometimes development versions are used.
# They may be written as 4.9.1-development
libversion_no_development = netCDF4.__netcdf4libversion__.split('-')[0]
libversion = tuple(int(v) for v in libversion_no_development.split('.'))
has_alignment = (libversion[0] > 4) or (
    libversion[0] == 4 and (libversion[1] >= 9)
)
try:
    has_h5ls = subprocess.check_call(['h5ls', '--version'], stdout=subprocess.PIPE) == 0
except Exception:
    has_h5ls = False

file_name = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name


class AlignmentTestCase(unittest.TestCase):
    def setUp(self):

        self.file = file_name

        # This is a global variable in netcdf4, it must be set before File
        # creation
        if has_alignment:
            set_alignment(1024, 4096)
            assert get_alignment() == (1024, 4096)

        f = Dataset(self.file, 'w')
        f.createDimension('x', 4096)
        # Create many datasets so that we decrease the chance of
        # the dataset being randomly aligned
        for i in range(10):
            f.createVariable(f'data{i:02d}', np.float64, ('x',))
            v = f.variables[f'data{i:02d}']
            v[...] = 0
        f.close()
        if has_alignment:
            # ensure to reset the alignment to 1 (default values) so as not to
            # disrupt other tests
            set_alignment(1, 1)
            assert get_alignment() == (1, 1)

    def test_version_settings(self):
        if has_alignment:
            # One should always be able to set the alignment to 1, 1
            set_alignment(1, 1)
            assert get_alignment() == (1, 1)
        else:
            with self.assertRaises(RuntimeError):
                set_alignment(1, 1)
            with self.assertRaises(RuntimeError):
                get_alignment()

    def test_reports_alignment_capabilities(self):
        # Assert that the library reports that it supports alignment correctly
        assert has_alignment == __has_set_alignment__

    # if we have no support for alignment, we have no guarantees on
    # how the data can be aligned
    @unittest.skipIf(
        not has_h5ls,
        "h5ls not found."
    )
    @unittest.skipIf(
        not has_alignment,
        "No support for set_alignment in libnetcdf."
    )
    def test_setting_alignment(self):
        # We choose to use h5ls instead of h5py since h5ls is very likely
        # to be installed alongside the rest of the tooling required to build
        # netcdf4-python
        # Output from h5ls is expected to look like:
        """
Opened "/tmp/tmpqexgozg1.nc" with sec2 driver.
data00                   Dataset {4096/4096}
    Attribute: DIMENSION_LIST {1}
        Type:      variable length of
                   object reference
    Attribute: _Netcdf4Coordinates {1}
        Type:      32-bit little-endian integer
    Location:  1:563
    Links:     1
    Storage:   32768 logical bytes, 32768 allocated bytes, 100.00% utilization
    Type:      IEEE 64-bit little-endian float
    Address:   8192
data01                   Dataset {4096/4096}
    Attribute: DIMENSION_LIST {1}
        Type:      variable length of
                   object reference
    Attribute: _Netcdf4Coordinates {1}
        Type:      32-bit little-endian integer
    Location:  1:1087
    Links:     1
    Storage:   32768 logical bytes, 32768 allocated bytes, 100.00% utilization
    Type:      IEEE 64-bit little-endian float
    Address:   40960
[...]
x                        Dataset {4096/4096}
    Attribute: CLASS scalar
        Type:      16-byte null-terminated ASCII string
    Attribute: NAME scalar
        Type:      64-byte null-terminated ASCII string
    Attribute: REFERENCE_LIST {10}
        Type:      struct {
                   "dataset"          +0    object reference
                   "dimension"        +8    32-bit little-endian unsigned integer
               } 16 bytes
    Attribute: _Netcdf4Dimid scalar
        Type:      32-bit little-endian integer
    Location:  1:239
    Links:     1
    Storage:   16384 logical bytes, 0 allocated bytes
    Type:      IEEE 32-bit big-endian float
    Address:   18446744073709551615
"""
        h5ls_results = subprocess.check_output(
            ["h5ls", "--verbose", "--address", "--simple", self.file]
        ).decode()

        addresses = {
            f'data{i:02d}': -1
            for i in range(10)
        }

        data_variable = None
        for line in h5ls_results.split('\n'):
            if not line.startswith(' '):
                data_variable = line.split(' ')[0]
            # only process the data variables we care to inpsect
            if data_variable not in addresses:
                continue
            line = line.strip()
            if line.startswith('Address:'):
                address = int(line.split(':')[1].strip())
                addresses[data_variable] = address

        for key, address in addresses.items():
            is_aligned = (address % 4096) == 0
            assert is_aligned, f"{key} is not aligned. Address = 0x{address:x}"

        # Alternative implementation in h5py
        # import h5py
        # with h5py.File(self.file, 'r') as h5file:
        #     for i in range(10):
        #         v = h5file[f'data{i:02d}']
        #         assert (dataset.id.get_offset() % 4096) == 0

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)


if __name__ == '__main__':
    unittest.main()
