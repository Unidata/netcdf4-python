import netCDF4, sys
# test variable creation.
import numpy as NP
import random

chars = '1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


FILE_NAME="tst_string.nc"
VAR_NAME="dummy_var"
DIM1_NAME="x"
DIM1_LEN=3
DIM2_NAME="y"
DIM2_LEN=3

data = NP.empty(DIM1_LEN*DIM2_LEN,'O')
for n in range(DIM1_LEN*DIM2_LEN):
    stringlen = random.randint(3,12)
    data[n] = ''.join([random.choice(chars) for i in range(stringlen)])
data[0] = {'spam':1,'eggs':2} # will be pickled into a string.
data = NP.reshape(data,(DIM1_LEN,DIM2_LEN))
print data.shape
print data.dtype.char
print data

f = netCDF4.Dataset(FILE_NAME,'w')
f.createDimension(DIM1_NAME, DIM1_LEN)
f.createDimension(DIM2_NAME, DIM2_LEN)
v = f.createVariable(VAR_NAME, 'S', (DIM1_NAME,DIM2_NAME))
v.long_name = 'dummy data'
v[:] = data
v[-1,-1] = 'hello'
print v.shape
print v.dtype
f.close()

f = netCDF4.Dataset(FILE_NAME,'r')
print f.variables
v = f.variables[VAR_NAME]
print v.long_name
print v.shape
print v.dtype
datout = v[:]
print datout.dtype.char
print datout.shape
print datout
f.close()
