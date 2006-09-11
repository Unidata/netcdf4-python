import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform 
from numpy.testing import assert_array_equal, assert_array_almost_equal
import netCDF4

def stringtoarr(string,NUMCHARS):
    """function to convert a string to a array of NUMCHARS characters"""
    arr = NP.zeros(NUMCHARS,'S1')
    arr[0:len(string)] = tuple(string)
    return arr

# test compound user-defined data type.
FILE_NAME = tempfile.mktemp(".nc")
# define a compound data type (a list of 3-tuples containing
# the name of each member, it's primitive data type, and it's size).
# Only fixed size primitive data types allowed (no 'S').
# Members can be multi-dimensional arrays (in which case the third
# element is a shape tuple instead of a scalar).
# create a user-defined data type
dtyp = [('latitude', 'f4', 1), ('longitude', 'f4', 1), ('sfc_press', 'i4', 1), ('temp_sounding', 'f4', (10,)), ('press_sounding', 'i4', (10,)), ('2dfield', 'f8', (4, 4)), ('location_name', 'S1', (80,))]
# create a record array of that dtype
rax = NP.empty(2,dtyp)
rax['latitude'] = 40.
rax['longitude'] = -105.
rax['sfc_press'] = 818
rax['temp_sounding'] = (280.3,272.,270.,269.,266.,258.,254.1,250.,245.5,240.)
rax['press_sounding'] = range(800,300,-50)
rax['2dfield'] = NP.ones((4,4),'f8')
# only fixed-size primitive data types can currenlty be used
# as compound data type members (although the library supports
# nested compound types).
# To store strings in a compound data type, each string must be stored as fixed-size
# (in this case 80) array of characters.
rax['location_name'] = stringtoarr('Boulder, Colorado, USA',80)
tupx =               (40.78,-73.99,1002,
                     (290.2,282.5,279.,277.9,276.,266.,264.1,260.,255.5,243.),
                     range(900,400,-50),2.*NP.ones((4,4),'f8'),
                     stringtoarr('New York, New York, USA',80))
units_dict = {'latitude': 'degrees north', 'longitude': 'degrees east', 'sfc_press': 'Pascals', 'temp_sounding': 'Kelvin', 'press_sounding': 'Pascals','location_name': None,'2dfield': 'Whatsits'}
rax[1] = tupx

class CompoundTestCase(unittest.TestCase):

    def setUp(self):
        # create a new file.
        self.file = FILE_NAME
        f = netCDF4.Dataset(FILE_NAME,'w')
        # create an unlimited  dimension call 'station'
        f.createDimension('station',False)
        table = f.createUserType(dtyp,'compound','station_data')
        # create a variable of this type.
        statdat = f.createVariable('station_obs', table, ('station',))
        # create a scalar variable of this type.
        statdat_scalar = f.createVariable('station_obs1', table)
        # assign record array to variable slice.
        statdat[0] = rax[0]
        # or just assign a tuple of values to variable slice
        # (will automatically be converted to a record array).
        statdat[1] = tupx
        statdat_scalar.assignValue(rax[0])
        # this module doesn't support attributes of compound type.
        # so, to assign an attribute like 'units' to each member of 
        # the compound type I do the following:
        # 1) create a python dict with key/value pairs representing
        #    the name of each compound type member and it's units.
        # 2) convert the dict to a string using the repr function.
        # 3) use that string as a variable attribute.
        # When this attribute is read back in it can be converted back to
        # a python dictionary using the eval function..
        # This can be converted into hash-like objects in other languages
        # as well (including C), since this string is also valid JSON
        # (JavaScript Object Notation - http://json.org). 
        # JSON is a lightweight, language-independent data serialization format.
        statdat.units = repr(units_dict)
        # close the file.
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing variables"""
        f  = netCDF4.Dataset(self.file, 'r')
        v = f.variables['station_obs']
        vs = f.variables['station_obs1']
        self.assert_(v.usertype == 'compound')
        dtyp = [('latitude', 'f4', 1), ('longitude', 'f4', 1), ('sfc_press', 'i4', 1), ('temp_sounding', 'f4', (10,)), ('press_sounding', 'i4', (10,)), ('2dfield', 'f8', (4, 4)), ('location_name', 'S1', (80,))]
        # check datatype, usertype name and shape
        self.assert_(v.dtype.base_datatype == dtyp)
        self.assert_(v.usertype_name == 'station_data')
        self.assert_(v.shape == (2,))
        units = eval(v.units)
        ra = v[:]
        # check record array returned by slice.
        self.assert_(ra.dtype.char == 'V')
        self.assert_(ra.dtype == dtyp)
        # check data in record array.
        for dt in dtyp:
            field = dt[0]
            for n in range(ra.shape[0]):
                self.assert_(units[field] == units_dict[field])
                if field != 'location_name':
                    assert_array_almost_equal(ra[n][field],rax[n][field])
                else:
                    self.assert_(ra[n][field].tostring() == rax[n][field].tostring())
        # check data in scalar variable.
        ra = vs.getValue()
        self.assert_(ra.dtype.char == 'V')
        self.assert_(ra.dtype == dtyp)
        for dt in dtyp:
            field = dt[0]
            self.assert_(units[field] == units_dict[field])
            if field != 'location_name':
                assert_array_almost_equal(ra[field],rax[0][field])
            else:
                self.assert_(ra[field].tostring() == rax[0][field].tostring())

        f.close()

if __name__ == '__main__':
    unittest.main()
