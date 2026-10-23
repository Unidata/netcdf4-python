import unittest, os
import numpy as np
from numpy.testing import assert_allclose
import netCDF4

# test reading a GRIB2 file with the NetCDF Expansion Pack (NEP) GRIB2
# reader (issue #1490). Only runs if the NEP GRIB2 reader is registered
# with netcdf-c (e.g. via NCRCENV_RC pointing to NEP's .ncrc).
# gfs_sample.grb2 is the first 3 messages of pygrib's sampledata/gfs.grb
# (10 hPa HGT, TMP and RH on a 2.5 degree global grid).

FILE_NAME = os.path.join(os.path.dirname(__file__), 'gfs_sample.grb2')
NLATS, NLONS = 73, 144

# reference values from pygrib: (value at [0,0], [36,72], [-1,-1], min, max)
REFERENCE = {
    'HGT': (29989.89, 31001.2, 27951.64, 27900.99, 31664.09),
    'TMP': (213.7, 233.1, 221.0, 209.3, 257.1),
    'RH': (0.0, 0.03, 0.06, 0.0, 0.21),
}

def _has_nep_grib2():
    try:
        netCDF4.Dataset(FILE_NAME).close()
        return True
    except OSError:
        return False

has_nep_grib2 = _has_nep_grib2()


@unittest.skipUnless(has_nep_grib2, "NEP GRIB2 reader not available")
class NEPGrib2TestCase(unittest.TestCase):

    def check(self, ds):
        assert ds.Conventions == 'GRIB2'
        assert ds.GRIB2_edition == 2
        assert list(ds.variables) == list(REFERENCE)
        for name, (v00, vmid, vlast, vmin, vmax) in REFERENCE.items():
            var = ds[name]
            assert var.long_name == name
            data = var[:]
            assert data.size == NLATS * NLONS
            # NEP currently defines the dimensions in the wrong order
            # (y=144, x=73), but the data are in the usual (lat, lon)
            # order, so reshape before comparing.
            data = data.reshape(NLATS, NLONS)
            assert_allclose([data[0, 0], data[NLATS//2, NLONS//2], data[-1, -1],
                             data.min(), data.max()],
                            [v00, vmid, vlast, vmin, vmax], rtol=1.e-5, atol=1.e-5)

    def runTest(self):
        """read GRIB2 file, with format detected by magic number"""
        with netCDF4.Dataset(FILE_NAME) as ds:
            self.check(ds)
        # selecting the GRIB2 reader explicitly gives the same result
        with netCDF4.Dataset(FILE_NAME, extra_mode_flag=netCDF4.NEP_UDF_GRIB2) as ds:
            self.check(ds)


if __name__ == '__main__':
    unittest.main()
