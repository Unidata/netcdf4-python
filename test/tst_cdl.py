import unittest
import netCDF4
import os

test_ncdump="""netcdf ubyte {
dimensions:
	d = 2 ;
variables:
	byte ub(d) ;
		ub:_Unsigned = "true" ;
	byte sb(d) ;

// global attributes:
		:_Format = "classic" ;
}
"""

class Test_CDL(unittest.TestCase):
    """
    Test import/export of CDL
    """
    def setUp(self):
        f=netCDF4.Dataset('ubyte.nc')
        f.tocdl(outfile='ubyte.cdl')
        f.close()
    def test_tocdl(self):
        # treated as unsigned integers.
        f=netCDF4.Dataset('ubyte.nc')
        assert(f.tocdl() == test_ncdump)
        f.close()
    def test_fromcdl(self):
        f1=netCDF4.Dataset.fromcdl('ubyte.cdl',ncfilename='ubyte2.nc')
        f2=netCDF4.Dataset('ubyte.nc')
        assert(f1.variables.keys() == f2.variables.keys())
        assert(f1.filepath() == 'ubyte2.nc')
        assert(f1.dimensions.keys() == f2.dimensions.keys())
        assert(len(f1.dimensions['d']) == len(f2.dimensions['d']))
        f1.close(); f2.close()
        os.remove('ubyte2.nc')
    def tearDown(self):
        # Remove the temporary files
        os.remove('ubyte.cdl')

if __name__ == '__main__':
    unittest.main()
