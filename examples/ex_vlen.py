import netCDF4
# test variable creation.
import numpy as NP

FILE_NAME="tst_vlen.nc"
VAR_NAME="dummy_var1"
VAR_NAME2="dummy_var2"
VLEN_NAME='vlen1'
DIM1_NAME="x"
DIM1_LEN=3
DIM2_NAME="y"
DIM2_LEN=3

data = NP.empty(DIM1_LEN*DIM2_LEN,'O')
for n in range(DIM1_LEN*DIM2_LEN):
    data[n] = NP.arange(n+1)+1
data = NP.reshape(data,(DIM1_LEN,DIM2_LEN))
print data.shape
print data

f = netCDF4.Dataset(FILE_NAME,'w')
f.createDimension(DIM1_NAME, DIM1_LEN)
f.createDimension(DIM2_NAME, DIM2_LEN)
vlen = f.createUserType('i4','vlen','vlen1')
v = f.createVariable(VAR_NAME, vlen, (DIM1_NAME,DIM2_NAME))
v2 = f.createVariable(VAR_NAME2, vlen, (DIM1_NAME,DIM2_NAME))
v.long_name = 'dummy data'
v[:] = data
v[-1,-1] = [-99,-98,-97]
#v[-1,-1] = NP.array([-99,-98,-97])
print v.usertype
print v.dtype.base_datatype
print v[:]
f.close()

f = netCDF4.Dataset(FILE_NAME,'r')
print f.variables
v = f.variables[VAR_NAME]
v2 = f.variables[VAR_NAME2]
print v.shape
print v.usertype
print v.dtype.base_datatype
print v.dtype.usertype_name
print v2.usertype
print v2.dtype.base_datatype
print v2.dtype.usertype_name
datout = v[:]
print datout.dtype.char
print datout
f.close()
