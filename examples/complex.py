from netCDF4 import Dataset
from numpy.random import uniform
import numpy as np
import os
# this example shows how to save a complex array using
# a compound data type.
f = Dataset('complex.nc','w')
size = 2 # length of 1-d complex array
# create random complex data.
datar = uniform(size=size)
datai = uniform(size=size)
datac = np.empty(size,np.complex128)
datac = datar + datai*1j
f.createDimension('phony_dim',None)
# create complex128 compound data type.
complex128 = np.dtype([('real',np.float64),('imag',np.float64)])
complex128_t = f.createCompoundType(complex128,'complex128')
# create a variable with this data type, write some data to it.
v = f.createVariable('phony_var',complex128_t,'phony_dim')
data = np.empty(len(datar),complex128)
data['real'] = datac.real
data['imag'] = datac.imag
v[:] = data
# close and reopen the file, check the contents.
f.close()
os.system('ncdump -h complex.nc')
f = Dataset('complex.nc')
v = f.variables['phony_var']
datain = v[:] # read in all the data into a numpy structured array
# create an empty numpy complex array
datac2 = np.empty(datain.shape,np.complex128)
# .. fill it with contents of structured array.
datac2.real = datain['real']
datac2.imag = datain['imag']
print 'original data:'
print datac.dtype,datac
print 'data read in from file:'
print datac2.dtype,datac2
f.close()
