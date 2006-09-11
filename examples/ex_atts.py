import netCDF4, sys
# test getting and setting global and variable attributes.
# (numbers, sequences and strings)
import numpy as NP
import math

FILE_NAME="tst_atts.nc"
VAR_DOUBLE_NAME="dummy_var"
GROUP_NAME = "dummy_group"
DIM1_NAME="x"
DIM1_LEN=2
DIM2_NAME="y"
DIM2_LEN=3
DIM3_NAME="z"
DIM3_LEN=25

f = netCDF4.Dataset(FILE_NAME,'w')
f.stratt = 'string attribute'
f.intatt = 1
f.floatatt = math.pi
f.seqatt = NP.arange(10)
# sequences of strings converted to a single string.
f.stringseqatt = ['mary ','had ','a ','little ','lamb']
# python objects that cannot be cast to numpy arrays are stored
# as pickled strings (and unpickled when accessed).
f.objatt = {'spam':1,'eggs':2}
g = f.createGroup(GROUP_NAME)
f.createDimension(DIM1_NAME, DIM1_LEN)
f.createDimension(DIM2_NAME, DIM2_LEN)
f.createDimension(DIM3_NAME, DIM3_LEN)
g.createDimension(DIM1_NAME, DIM1_LEN)
g.createDimension(DIM2_NAME, DIM2_LEN)
g.createDimension(DIM3_NAME, DIM3_LEN)
g.stratt = 'string attribute'
g.intatt = 1
g.floatatt = math.pi
g.seqatt = NP.arange(10)
g.stringseqatt = ['mary ','had ','a ','little ','lamb']
v = f.createVariable(VAR_DOUBLE_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
v.stratt = 'string attribute'
v.intatt = 1
v.floatatt = math.pi
v.seqatt = NP.arange(10)
v.stringseqatt = ['mary ','had ','a ','little ','lamb']
v.objatt = {'spam':1,'eggs':2}
v1 = g.createVariable(VAR_DOUBLE_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
v1.stratt = 'string attribute'
v1.intatt = 1
v1.floatatt = math.pi
v1.seqatt = NP.arange(10)
v1.stringseqatt = ['mary ','had ','a ','little ','lamb']
v1.objatt = {'spam':1,'eggs':2}
f.close()

f = netCDF4.Dataset(FILE_NAME,'r')
print 'root group attributes'
for name in f.ncattrs():
    print 'Global attr for root group', name, '=', getattr(f,name)
print 'group %s attributes' % GROUP_NAME
grp = f.groups[GROUP_NAME]
for name in grp.ncattrs():
    print 'Global attr for %s' % GROUP_NAME, name, '=', getattr(f,name)
print 'variable %s (in root group) attributes' % VAR_DOUBLE_NAME
for name in f.variables[VAR_DOUBLE_NAME].ncattrs():
    print 'Var attr for %s in root group' % VAR_DOUBLE_NAME, name, '=', getattr(f,name)
print 'variable %s in %s attributes' % (VAR_DOUBLE_NAME,GROUP_NAME)
for name in grp.variables[VAR_DOUBLE_NAME].ncattrs():
    print 'Var attr for %s in %s' % (VAR_DOUBLE_NAME,GROUP_NAME), name, '=', getattr(f,name)
f.close()
