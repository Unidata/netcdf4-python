import netCDF4, sys
# test variable creation.
from numpy.random.mtrand import uniform
import numpy as NP

FILE_NAME="tst_vars.nc"
VAR_DOUBLE_NAME="dummy_var"
GROUP_NAME = "dummy_group"
DIM1_NAME="x"
DIM1_LEN=2
DIM2_NAME="y"
DIM2_LEN=3
DIM3_NAME="z"
DIM3_LEN=25


randomdata = uniform(size=(DIM1_LEN,DIM2_LEN,DIM3_LEN))
print randomdata[::2,:,-1]

f = netCDF4.Dataset(FILE_NAME,'w')
f.createDimension(DIM1_NAME, DIM1_LEN)
f.createDimension(DIM2_NAME, DIM2_LEN)
f.createDimension(DIM3_NAME, DIM3_LEN)
v = f.createVariable(VAR_DOUBLE_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
g = f.createGroup(GROUP_NAME)
g.createDimension(DIM1_NAME, DIM1_LEN)
g.createDimension(DIM2_NAME, DIM2_LEN)
g.createDimension(DIM3_NAME, DIM3_LEN)
v1 = g.createVariable(VAR_DOUBLE_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
v.long_name = 'dummy data'
v[:] = randomdata
v1[:] = randomdata
f.close()

f = netCDF4.Dataset(FILE_NAME,'r')
print f.variables
v = f.variables[VAR_DOUBLE_NAME]
print f.groups[GROUP_NAME].variables
v1 = f.groups[GROUP_NAME].variables[VAR_DOUBLE_NAME]
print v.shape
print v1.shape
datout = v[::2,:,-1]
datout2 = v[::2,:,-1]
print datout.shape
print datout
print datout2.shape
print datout2
f.close()

