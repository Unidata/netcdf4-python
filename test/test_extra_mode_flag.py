import unittest, os, tempfile
from packaging.version import Version
from numpy.testing import assert_array_almost_equal
import netCDF4

# test extra_mode_flag Dataset kwarg (issue #1490), used to select a
# user-defined format (UDF) dispatch layer for files with no magic number.
# The read tests only run if the NetCDF Expansion Pack (NEP) mmCIF reader
# is registered with netcdf-c (e.g. via NCRCENV_RC pointing to NEP's .ncrc).

MMCIF = """\
data_TINY
#
_entry.id TINY
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . VAL A 1 1 ? 6.204 16.869 4.854 1.00 49.05 1 A 1
ATOM 2 C CA . VAL A 1 1 ? 6.913 17.759 4.607 1.00 43.14 1 A 1
ATOM 3 C C . VAL A 1 1 ? 8.504 17.378 4.797 1.00 24.80 1 A 1
#
"""
CARTN_X = [6.204, 6.913, 8.504]

has_udf = Version(netCDF4.__netcdf4libversion__.split('-')[0]) >= Version('4.10.0')

def _write(text):
    f = tempfile.NamedTemporaryFile(suffix='.cif', delete=False, mode='w')
    f.write(text)
    f.close()
    return f.name

def _has_nep_mmcif():
    if not has_udf:
        return False
    filename = _write(MMCIF)
    try:
        netCDF4.Dataset(filename).close()
        return True
    except OSError:
        return False
    finally:
        os.remove(filename)

has_nep_mmcif = _has_nep_mmcif()


@unittest.skipUnless(has_udf, "extra_mode_flag requires netcdf-c >= 4.10.0")
class ExtraModeFlagArgsTestCase(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
        netCDF4.Dataset(self.file, 'w').close()

    def tearDown(self):
        os.remove(self.file)

    def runTest(self):
        """only allowed for plain read-only opens with a NEP_UDF_* flag"""
        flag = netCDF4.NEP_UDF_MMCIF
        for kwargs in (dict(mode='w'), dict(mode='a'), dict(mode='r+'),
                       dict(mode='rs'), dict(mode='r', diskless=True)):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                netCDF4.Dataset(self.file, extra_mode_flag=flag, **kwargs)
        with open(self.file, 'rb') as f:
            buf = f.read()
        with self.assertRaises(ValueError):
            netCDF4.Dataset('inmemory.nc', memory=buf, extra_mode_flag=flag)
        # not a NEP_UDF_* flag (NC_WRITE)
        with self.assertRaises(ValueError):
            netCDF4.Dataset(self.file, extra_mode_flag=0x0001)
        # extra_mode_flag=None is the default
        netCDF4.Dataset(self.file, extra_mode_flag=None).close()


@unittest.skipUnless(has_nep_mmcif, "NEP mmCIF reader not available")
class ExtraModeFlagNEPTestCase(unittest.TestCase):

    def setUp(self):
        # without the leading data_ magic number netcdf-c can't detect the format
        self.file = _write("# no magic number\n" + MMCIF)

    def tearDown(self):
        os.remove(self.file)

    def runTest(self):
        """open mmCIF file with no magic number using NEP_UDF_MMCIF"""
        with self.assertRaises(OSError):
            netCDF4.Dataset(self.file)
        with netCDF4.Dataset(self.file, extra_mode_flag=netCDF4.NEP_UDF_MMCIF) as ds:
            assert ds.entry_id == 'TINY'
            assert len(ds.dimensions['atom']) == 3
            assert_array_almost_equal(ds['atom_site_Cartn_x'][0], CARTN_X)


if __name__ == '__main__':
    unittest.main()
