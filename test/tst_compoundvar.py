import sys
import unittest
import os
import tempfile
from netCDF4 import Dataset, CompoundType
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

# test compound data types.

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
DIM_NAME = 'phony_dim'
GROUP_NAME = 'phony_group'
VAR_NAME = 'phony_compound_var'
TYPE_NAME1 = 'cmp1'
TYPE_NAME2 = 'cmp2'
TYPE_NAME3 = 'cmp3'
TYPE_NAME4 = 'cmp4'
TYPE_NAME5 = 'cmp5'
DIM_SIZE=3
# unaligned data types (note they are nested)
dtype1=np.dtype([('i', 'i2'), ('j', 'i8')])
dtype2=np.dtype([('x', 'f4',), ('y', 'f8',(3,2))])
dtype3=np.dtype([('xx', dtype1), ('yy', dtype2)])
dtype4=np.dtype([('xxx',dtype3),('yyy','f8', (4,))])
dtype5=np.dtype([('x1', dtype1), ('y1', dtype2)])
# aligned data types
dtype1a = np.dtype({'names':['i','j'],'formats':['<i2','<i8']},align=True)
dtype2a = np.dtype({'names':['x','y'],'formats':['<f4',('<f8', (3, 2))]},align=True)
dtype3a = np.dtype({'names':['xx','yy'],'formats':[dtype1a,dtype2a]},align=True)
dtype4a = np.dtype({'names':['xxx','yyy'],'formats':[dtype3a,('f8', (4,))]},align=True)
dtype5a = np.dtype({'names':['x1','y1'],'formats':[dtype1a,dtype2a]},align=True)

data = np.zeros(DIM_SIZE,dtype4)
data['xxx']['xx']['i']=1
data['xxx']['xx']['j']=2
data['xxx']['yy']['x']=3
data['xxx']['yy']['y']=4
data['yyy'] = 5
datag = np.zeros(DIM_SIZE,dtype5)
datag['x1']['i']=10
datag['x1']['j']=20
datag['y1']['x']=30
datag['y1']['y']=40

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = Dataset(self.file, 'w')
        d = f.createDimension(DIM_NAME,DIM_SIZE)
        g = f.createGroup(GROUP_NAME)
        # simple compound types.
        cmptype1 = f.createCompoundType(dtype1, TYPE_NAME1)
        cmptype2 = f.createCompoundType(dtype2, TYPE_NAME2)
        # close and reopen the file to make sure compound
        # type info read back in correctly.
        f.close()
        f = Dataset(self.file,'r+')
        g = f.groups[GROUP_NAME]
        # multiply nested compound types
        cmptype3 = f.createCompoundType(dtype3, TYPE_NAME3)
        cmptype4 = f.createCompoundType(dtype4, TYPE_NAME4)
        cmptype5 = f.createCompoundType(dtype5, TYPE_NAME5)
        v = f.createVariable(VAR_NAME,cmptype4, DIM_NAME)
        vv = g.createVariable(VAR_NAME,cmptype5, DIM_NAME)
        v[:] = data
        vv[:] = datag
        # try reading the data back before the file is closed
        dataout = v[:]
        dataoutg = vv[:]
        assert (cmptype4 == dtype4a) # data type should be aligned
        assert (dataout.dtype == dtype4a) # data type should be aligned
        assert(list(f.cmptypes.keys()) ==\
               [TYPE_NAME1,TYPE_NAME2,TYPE_NAME3,TYPE_NAME4,TYPE_NAME5])
        assert_array_equal(dataout['xxx']['xx']['i'],data['xxx']['xx']['i'])
        assert_array_equal(dataout['xxx']['xx']['j'],data['xxx']['xx']['j'])
        assert_array_almost_equal(dataout['xxx']['yy']['x'],data['xxx']['yy']['x'])
        assert_array_almost_equal(dataout['xxx']['yy']['y'],data['xxx']['yy']['y'])
        assert_array_almost_equal(dataout['yyy'],data['yyy'])
        assert_array_equal(dataoutg['x1']['i'],datag['x1']['i'])
        assert_array_equal(dataoutg['x1']['j'],datag['x1']['j'])
        assert_array_almost_equal(dataoutg['y1']['x'],datag['y1']['x'])
        assert_array_almost_equal(dataoutg['y1']['y'],datag['y1']['y'])
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)
        #pass

    def runTest(self):
        """testing compound variables"""
        f = Dataset(self.file, 'r')
        v = f.variables[VAR_NAME]
        g = f.groups[GROUP_NAME]
        vv = g.variables[VAR_NAME]
        dataout = v[:]
        dataoutg = vv[:]
        # make sure data type is aligned
        assert (f.cmptypes['cmp4'] == dtype4a)
        assert(list(f.cmptypes.keys()) ==\
               [TYPE_NAME1,TYPE_NAME2,TYPE_NAME3,TYPE_NAME4,TYPE_NAME5])
        assert_array_equal(dataout['xxx']['xx']['i'],data['xxx']['xx']['i'])
        assert_array_equal(dataout['xxx']['xx']['j'],data['xxx']['xx']['j'])
        assert_array_almost_equal(dataout['xxx']['yy']['x'],data['xxx']['yy']['x'])
        assert_array_almost_equal(dataout['xxx']['yy']['y'],data['xxx']['yy']['y'])
        assert_array_almost_equal(dataout['yyy'],data['yyy'])
        assert_array_equal(dataoutg['x1']['i'],datag['x1']['i'])
        assert_array_equal(dataoutg['x1']['j'],datag['x1']['j'])
        assert_array_almost_equal(dataoutg['y1']['x'],datag['y1']['x'])
        assert_array_almost_equal(dataoutg['y1']['y'],datag['y1']['y'])
        f.close()
        # issue 773
        f = Dataset(self.file,'w')
        dtype = np.dtype([('observation', 'i4'),
                          ('station_name','S80')])
        dtype_nest = np.dtype([('observation', 'i4'),
                               ('station_name','S80'),
                               ('nested_observation',dtype)])
        station_data_t1 = f.createCompoundType(dtype,'station_data1')
        station_data_t2 = f.createCompoundType(dtype_nest,'station_data')
        f.createDimension('station',None)
        statdat = f.createVariable('station_obs', station_data_t2, ('station',))
        assert(statdat.dtype == station_data_t2.dtype)
        datain = np.empty(2,station_data_t2.dtype_view)
        datain['observation'][:] = (123,314)
        datain['station_name'][:] = ('Boulder','New York')
        datain['nested_observation']['observation'][:] = (-999,999)
        datain['nested_observation']['station_name'][:] = ('Boston','Chicago')
        statdat[:] = datain
        f.close()
        f = Dataset(self.file)
        dataout = f['station_obs'][:]
        assert(dataout.dtype == station_data_t2.dtype_view)
        assert_array_equal(datain, dataout)
        f.close()

if __name__ == '__main__':
    from netCDF4 import getlibversion
    version =  getlibversion().split()[0]
    unittest.main()
