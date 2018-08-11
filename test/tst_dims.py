import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform
import netCDF4

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
LAT_NAME="lat"
LAT_LEN = 25
LAT_LENG = 50
LON_NAME="lon"
LON_LEN = 50
LON_LENG = 100
LEVEL_NAME="level"
LEVEL_LEN = None
LEVEL_LENG = None
TIME_NAME="time"
TIME_LEN = None
TIME_LENG = None
GROUP_NAME='forecasts'
VAR_NAME='temp'
VAR_TYPE='f8'


class DimensionsTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = netCDF4.Dataset(self.file, 'w')
        f.createDimension(LAT_NAME,LAT_LEN)
        f.createDimension(LON_NAME,LON_LEN)
        f.createDimension(LEVEL_NAME,LEVEL_LEN)
        f.createDimension(TIME_NAME,TIME_LEN)
        f.createVariable(VAR_NAME,VAR_TYPE,(LEVEL_NAME, LAT_NAME, LON_NAME, TIME_NAME))
        g = f.createGroup(GROUP_NAME)
        g.createDimension(LAT_NAME,LAT_LENG)
        g.createDimension(LON_NAME,LON_LENG)
        # should get dimensions from parent group.
        # (did not work prior to alpha 18)
        #g.createDimension(LEVEL_NAME,LEVEL_LENG)
        #g.createDimension(TIME_NAME,TIME_LENG)
        g.createVariable(VAR_NAME,VAR_TYPE,(LEVEL_NAME, LAT_NAME, LON_NAME, TIME_NAME))
        f.close()

    def tearDown(self):
        # Remove the temporary file
        os.remove(self.file)

    def runTest(self):
        """testing dimensions"""
        # check dimensions in root group.
        f  = netCDF4.Dataset(self.file, 'r+')
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
            self.assertTrue(name in names_check)
        # check that dimension lengths are correct.
        for name,dim in f.dimensions.items():
            self.assertTrue(len(dim) == lensdict[name])
        # check that isunlimited() method works.
        for name,dim in f.dimensions.items():
            self.assertTrue(dim.isunlimited() == unlimdict[name])
        # add some data to variable along unlimited dims,
        # make sure length of dimensions change correctly.
        nadd1 = 2
        nadd2 = 4
        v[0:nadd1,:,:,0:nadd2] = uniform(size=(nadd1,LAT_LEN,LON_LEN,nadd2))
        lensdict[LEVEL_NAME]=nadd1
        lensdict[TIME_NAME]=nadd2
        # check that dimension lengths are correct.
        for name,dim in f.dimensions.items():
            self.assertTrue(len(dim) == lensdict[name])
        # check dimensions in subgroup.
        g = f.groups[GROUP_NAME]
        vg = g.variables[VAR_NAME]
        isunlim = [dim.isunlimited() for dim in g.dimensions.values()]
        dimlens = [len(dim) for dim in g.dimensions.values()]
        names_check = [LAT_NAME, LON_NAME, LEVEL_NAME, TIME_NAME]
        lens_check = [LAT_LENG, LON_LENG, LEVEL_LENG, TIME_LENG]
        isunlim = [dimlen == None for dimlen in lens_check]
        for n,dimlen in enumerate(lens_check):
            if dimlen is None:
                lens_check[n] = 0
        lensdict = dict(zip(names_check,lens_check))
        unlimdict = dict(zip(names_check,isunlim))
        # check that dimension names are correct.
        for name in g.dimensions.keys():
            self.assertTrue(name in names_check)
        # check that dimension lengths are correct.
        for name,dim in g.dimensions.items():
            self.assertTrue(len(dim) == lensdict[name])
        # check get_dims variable method
        dim_tuple = vg.get_dims()
        # some dimensions from parent group
        dim_tup1 = (f.dimensions['level'],g.dimensions['lat'],\
                    g.dimensions['lon'],f.dimensions['time'])
        dim_tup2 = vg.get_dims()
        assert(dim_tup1 == dim_tup2)
        # check that isunlimited() method works.
        for name,dim in g.dimensions.items():
            self.assertTrue(dim.isunlimited() == unlimdict[name])
        # add some data to variable along unlimited dims,
        # make sure length of dimensions change correctly.
        nadd1 = 8
        nadd2 = 4
        vg[0:nadd1,:,:,0:nadd2] = uniform(size=(nadd1,LAT_LENG,LON_LENG,nadd2))
        lensdict[LEVEL_NAME]=nadd1
        lensdict[TIME_NAME]=nadd2
        for name,dim in g.dimensions.items():
            self.assertTrue(len(dim) == lensdict[name])
        f.close()

if __name__ == '__main__':
    unittest.main()
