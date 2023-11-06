import unittest
import netCDF4
import os
import pathlib

test_ncdump="""netcdf ubyte {
dimensions:
	d = 2 ;
variables:
	byte ub(d) ;
		ub:_Unsigned = "true" ;
	byte sb(d) ;
	byte sb2(d) ;
		sb2:_Unsigned = "false" ;

// global attributes:
		:_Format = "classic" ;
}
"""
test_ncdump2="""netcdf ubyte {
dimensions:
	d = 2 ;
variables:
	byte ub(d) ;
		ub:_Unsigned = "true" ;
	byte sb(d) ;
	byte sb2(d) ;
		sb2:_Unsigned = "false" ;

// global attributes:
		:_Format = "classic" ;
data:

 ub = 0, -1 ;

 sb = -128, 127 ;

 sb2 = -127, -127 ;
}
"""


ubyte_filename = pathlib.Path(__file__).parent / "ubyte.nc"


@unittest.skipIf(os.getenv("NO_CDL"), "CDL test disabled")
class Test_CDL(unittest.TestCase):
    """
    Test import/export of CDL
    """

    def setUp(self):
        with netCDF4.Dataset(ubyte_filename) as f:
            f.tocdl(outfile="ubyte.cdl", data=True)

    def test_tocdl(self):
        # treated as unsigned integers.
        with netCDF4.Dataset(ubyte_filename) as f:
            assert f.tocdl() == test_ncdump
            assert f.tocdl(data=True) == test_ncdump2

    def test_fromcdl(self):
        with netCDF4.Dataset.fromcdl("ubyte.cdl", ncfilename="ubyte2.nc") as f1:
            with netCDF4.Dataset(ubyte_filename) as f2:
                assert f1.variables.keys() == f2.variables.keys()
                assert f1.filepath() == "ubyte2.nc"
                assert f1.dimensions.keys() == f2.dimensions.keys()
                assert len(f1.dimensions["d"]) == len(f2.dimensions["d"])
                assert (f1["ub"][:] == f2["ub"][:]).all()
                assert (f1["sb"][:] == f2["sb"][:]).all()

        os.remove("ubyte2.nc")

    def tearDown(self):
        # Remove the temporary files
        os.remove('ubyte.cdl')

if __name__ == '__main__':
    unittest.main()
