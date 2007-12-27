import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform 
import netCDF3

FILE_NAME = tempfile.mktemp(".nc")
LAT_NAME="lat"
LAT_LEN = 25
LON_NAME="lon"
LON_LEN = 50  
LEVEL_NAME="level"
LEVEL_LEN = 10
TIME_NAME="time"
TIME_LEN = None
VAR_NAME='temp'
VAR_TYPE='f8'


class DimensionsTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = netCDF3.Dataset(self.file, 'w')
        f.createDimension(LAT_NAME,LAT_LEN)
        f.createDimension(LON_NAME,LON_LEN)
        f.createDimension(LEVEL_NAME,LEVEL_LEN)
        f.createDimension(TIME_NAME,TIME_LEN)
        f.createVariable(VAR_NAME,VAR_TYPE,(TIME_NAME, LAT_NAME, LON_NAME, LEVEL_NAME))
        f.close()

    def tearDown(self):
        # Remove the temporary file
        os.remove(self.file)

    def runTest(self):
        """testing dimensions"""
        # check dimensions in root group.
        f  = netCDF3.Dataset(self.file, 'r+')
        v = f.variables[VAR_NAME]
        isunlim = [dim.isunlimited() for dim in f.dimensions.values()]
        dimlens = [len(dim) for dim in f.dimensions.values()]
        names_check = [LAT_NAME, LON_NAME, LEVEL_NAME, TIME_NAME]
        lens_check = [LAT_LEN, LON_LEN, LEVEL_LEN, TIME_LEN]
        isunlim = [dimlen == None for dimlen in lens_check]
        for n,dimlen in enumerate(lens_check):
            if dimlen is None:
                lens_check[n] = 0
        lensdict = dict(zip(names_check,lens_check))
        unlimdict = dict(zip(names_check,isunlim))
        # check that dimension names are correct.
        for name in f.dimensions.keys():
            self.assert_(name in names_check)
        # check that dimension lengths are correct.
        for name,dim in f.dimensions.iteritems():
            self.assert_(len(dim) == lensdict[name])
        # check that isunlimited() method works.
        for name,dim in f.dimensions.iteritems():
            self.assert_(dim.isunlimited() == unlimdict[name])
        # add some data to variable along unlimited dims,
        # make sure length of dimensions change correctly.
        nadd = 5
        v[0:nadd,:,:,:] = uniform(size=(nadd,LAT_LEN,LON_LEN,LEVEL_LEN))
        lensdict[TIME_NAME]=nadd
        # check that dimension lengths are correct.
        for name,dim in f.dimensions.iteritems():
            self.assert_(len(dim) == lensdict[name])
        f.close()

if __name__ == '__main__':
    unittest.main()
