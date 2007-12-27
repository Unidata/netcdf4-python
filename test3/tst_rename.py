import sys
import unittest
import os
import tempfile
import numpy as NP
import netCDF3

# test changing dimension, variable names
# and deleting attributes.

FILE_NAME = tempfile.mktemp(".nc")
LAT_NAME="lat"
LON_NAME="lon"
LEVEL_NAME="level"
TIME_NAME="time"
VAR_NAME='temp'
LON_NAME2 = "longitude"
VAR_NAME2='wind'

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = netCDF3.Dataset(self.file, 'w')
        f.createDimension(LAT_NAME,73)
        f.createDimension(LON_NAME,145)
        f.createDimension(LEVEL_NAME,10)
        f.createDimension(TIME_NAME,None)
        f.foo = 'bar'
        f.goober = 2
        f.createVariable(VAR_NAME,'f4',(TIME_NAME, LAT_NAME, LON_NAME))
        v = f.variables[VAR_NAME]
        v.bar = 'foo'
        v.slobber = 3
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing renaming of dimensions, variables and attribute deletion"""
        f  = netCDF3.Dataset(self.file, 'r+')
        v = f.variables[VAR_NAME]
        names_check = [LAT_NAME, LON_NAME, LEVEL_NAME, TIME_NAME]
        # check that dimension names are correct.
        for name in f.dimensions.keys():
            self.assert_(name in names_check)
        names_check = [VAR_NAME]
        # check that variable names are correct.
        for name in f.variables.keys():
            self.assert_(name in names_check)
        # rename dimension.
        f.renameDimension(LON_NAME,LON_NAME2)
        # rename variable.
        f.renameVariable(VAR_NAME,VAR_NAME2)
        # check that new dimension names are correct.
        names_check = [LAT_NAME, LON_NAME2, LEVEL_NAME, TIME_NAME]
        for name in f.dimensions.keys():
            self.assert_(name in names_check)
        names_check = [VAR_NAME2]
        # check that new variable names are correct.
        for name in f.variables.keys():
            self.assert_(name in names_check)
        # delete a global attribute.
        atts = f.ncattrs()
        del f.goober
        atts.remove('goober')
        self.assert_(atts == f.ncattrs())
        # delete a variable attribute.
        atts = v.ncattrs()
        del v.slobber
        atts.remove('slobber')
        self.assert_(atts == v.ncattrs())
        f.close()
        # make sure attributes cannot be deleted, or vars/dims renamed
        # when file is open read-only.
        f  = netCDF3.Dataset(self.file)
        v = f.variables[VAR_NAME2]
        self.assertRaises(RuntimeError, delattr, v, 'bar')
        self.assertRaises(RuntimeError, f.renameVariable, VAR_NAME2, VAR_NAME)
        self.assertRaises(RuntimeError, f.renameDimension, LON_NAME2, LON_NAME)
        f.close()

if __name__ == '__main__':
    unittest.main()
