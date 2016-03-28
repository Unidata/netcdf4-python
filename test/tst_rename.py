import sys
import unittest
import os
import tempfile
import numpy as NP
import netCDF4
from netCDF4 import __has_rename_grp__

# test changing dimension, variable names
# and deleting attributes.

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
LAT_NAME="lat"
LON_NAME="lon"
LON_NAME2 = "longitude"
LEVEL_NAME="level"
TIME_NAME="time"
VAR_NAME='temp'
VAR_NAME2='wind'
GROUP_NAME='subgroup'
GROUP_NAME2='subgroup2'

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = netCDF4.Dataset(self.file, 'w')
        f.createDimension(LAT_NAME,73)
        f.createDimension(LON_NAME,145)
        f.createDimension(LEVEL_NAME,10)
        f.createDimension(TIME_NAME,None)
        if __has_rename_grp__:
            g = f.createGroup(GROUP_NAME)
        else:
            g = f.createGroup(GROUP_NAME2)
        g.createDimension(LAT_NAME,145)
        g.createDimension(LON_NAME,289)
        g.createDimension(LEVEL_NAME,20)
        g.createDimension(TIME_NAME,None)
        f.foo = 'bar'
        f.goober = 2
        g.foo = 'bar'
        g.goober = 2
        f.createVariable(VAR_NAME,'f4',(LAT_NAME, LON_NAME, TIME_NAME))
        v = f.variables[VAR_NAME]
        v.bar = 'foo'
        v.slobber = 3
        g.createVariable(VAR_NAME,'f4',(LAT_NAME, LON_NAME, TIME_NAME))
        v2 = g.variables[VAR_NAME]
        v2.bar = 'foo'
        v2.slobber = 3
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing renaming of dimensions, variables and attribute deletion"""
        f  = netCDF4.Dataset(self.file, 'r+')
        v = f.variables[VAR_NAME]
        names_check = [LAT_NAME, LON_NAME, LEVEL_NAME, TIME_NAME]
        # check that dimension names are correct.
        for name in f.dimensions.keys():
            self.assertTrue(name in names_check)
        names_check = [VAR_NAME]
        # check that variable names are correct.
        for name in f.variables.keys():
            self.assertTrue(name in names_check)
        # rename dimension.
        f.renameDimension(LON_NAME,LON_NAME2)
        # rename variable.
        f.renameVariable(VAR_NAME,VAR_NAME2)
        # rename group.
        if __has_rename_grp__:
            f.renameGroup(GROUP_NAME,GROUP_NAME2)
        # check that new dimension names are correct.
        names_check = [LAT_NAME, LON_NAME2, LEVEL_NAME, TIME_NAME]
        for name in f.dimensions.keys():
            self.assertTrue(name in names_check)
        names_check = [VAR_NAME2]
        # check that new variable names are correct.
        for name in f.variables.keys():
            self.assertTrue(name in names_check)
        g = f.groups[GROUP_NAME2]
        vg = g.variables[VAR_NAME]
        names_check = [LAT_NAME, LON_NAME, LEVEL_NAME, TIME_NAME]
        # check that dimension names are correct.
        for name in g.dimensions.keys():
            self.assertTrue(name in names_check)
        names_check = [VAR_NAME]
        # check that variable names are correct.
        for name in g.variables.keys():
            self.assertTrue(name in names_check)
        # check that group name is correct.
        self.assertTrue(GROUP_NAME not in f.groups and GROUP_NAME2 in f.groups)
        # rename dimension.
        g.renameDimension(LON_NAME,LON_NAME2)
        # rename variable.
        g.renameVariable(VAR_NAME,VAR_NAME2)
        # check that new dimension names are correct.
        names_check = [LAT_NAME, LON_NAME2, LEVEL_NAME, TIME_NAME]
        for name in g.dimensions.keys():
            self.assertTrue(name in names_check)
        names_check = [VAR_NAME2]
        # check that new variable names are correct.
        for name in g.variables.keys():
            self.assertTrue(name in names_check)
        # delete a global attribute.
        atts = f.ncattrs()
        del f.goober
        atts.remove('goober')
        self.assertTrue(atts == f.ncattrs())
        atts = g.ncattrs()
        del g.goober
        atts.remove('goober')
        self.assertTrue(atts == g.ncattrs())
        # delete a variable attribute.
        atts = v.ncattrs()
        del v.slobber
        atts.remove('slobber')
        self.assertTrue(atts == v.ncattrs())
        atts = vg.ncattrs()
        del vg.slobber
        atts.remove('slobber')
        self.assertTrue(atts == vg.ncattrs())
        f.close()
        # make sure attributes cannot be deleted, or vars/dims renamed
        # when file is open read-only.
        f  = netCDF4.Dataset(self.file)
        v = f.variables[VAR_NAME2]
        self.assertRaises(RuntimeError, delattr, v, 'bar')
        self.assertRaises(RuntimeError, f.renameVariable, VAR_NAME2, VAR_NAME)
        self.assertRaises(RuntimeError, f.renameDimension, LON_NAME2, LON_NAME)
        g = f.groups[GROUP_NAME2]
        vg = g.variables[VAR_NAME2]
        self.assertRaises(RuntimeError, delattr, vg, 'bar')
        self.assertRaises(RuntimeError, g.renameVariable, VAR_NAME2, VAR_NAME)
        self.assertRaises(RuntimeError, g.renameDimension, LON_NAME2, LON_NAME)
        f.close()

if __name__ == '__main__':
    unittest.main()
